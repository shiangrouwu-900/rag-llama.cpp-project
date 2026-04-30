# rag/evaluation.py
import json
import os
from datetime import datetime
from pathlib import Path

from rag.build_index import main as build_index_main
from rag.embedding import load_embedding_model
from rag.generation import build_prompt, generate_stream, load_llm
from rag.retrieval import load_index, retrieve


TOP_K = 3
MODEL_NAME = "qwen2.5-1.5b.q4_k_m"
MODEL_PATH = os.getenv("MODEL_PATH")
N_GPU_LAYERS = 20

TEST_DATA_PATH = "data/test_data.json"
INDEX_DIR = "storage"
OUTPUT_PATH = "storage/evaluation_records.jsonl"
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "0") == "1"


def load_test_data(path=TEST_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(record, path=OUTPUT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_index(index_dir=INDEX_DIR, rebuild=REBUILD_INDEX):
    """
    evaluation.py 不再自行 build chunks / embeddings。
    需要 index 時，統一交給 build_index.py 建立，再由 retrieval.py 載入。
    """
    chunks_path = Path(index_dir) / "chunks.json"
    embeddings_path = Path(index_dir) / "embeddings.npy"

    if rebuild or not chunks_path.exists() or not embeddings_path.exists():
        print("找不到 index，或指定 REBUILD_INDEX=1，開始執行 build_index.py...")
        build_index_main()

    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"index 建立失敗，請確認 {chunks_path} 與 {embeddings_path} 是否存在。"
        )


def validate_model_path(model_path):
    if not model_path:
        raise ValueError(
            "MODEL_PATH 尚未設定。請先設定環境變數，例如：\n"
            "export MODEL_PATH=/content/drive/MyDrive/rag_models/qwen2.5-1.5b.q4_k_m.gguf"
        )

    if not Path(model_path).exists():
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")


def main():
    validate_model_path(MODEL_PATH)

    print("確認 / 建立 index...")
    ensure_index(INDEX_DIR)

    print("載入 index...")
    embeddings, chunks = load_index(INDEX_DIR)
    print(f"Chunks 數量: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")

    print("載入 embedding model...")
    embedding_model = load_embedding_model()

    print("載入 LLM...")
    llm = load_llm(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
    )

    test_data = load_test_data(TEST_DATA_PATH)

    print(f"開始 evaluation，共 {len(test_data)} 題。")
    print(f"結果會儲存到：{OUTPUT_PATH}")

    for i, item in enumerate(test_data, start=1):
        query = item["question"]

        print("\n" + "=" * 80)
        print(f"[{i}/{len(test_data)}] Query: {query}")
        print("=" * 80)

        results = retrieve(
            query=query,
            model=embedding_model,
            embeddings=embeddings,
            chunks=chunks,
            top_k=TOP_K,
        )

        prompt = build_prompt(query, results)

        print("回答：")
        answer, metrics = generate_stream(llm, prompt)
        print()

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "expected_answer": item.get("expected_answer"),
            "type": item.get("type"),
            "answer": answer.strip(),
            "ttft": metrics["ttft"],
            "tps": metrics["tps"],
            "output_tokens": metrics["output_tokens"],
            "top_k": TOP_K,
            "model": MODEL_NAME,
            "model_path": MODEL_PATH,
            "n_gpu_layers": N_GPU_LAYERS,
        }

        save_jsonl(record)

        print("metrics:")
        print(json.dumps(record, ensure_ascii=False, indent=2))

    print("\nEvaluation 完成。")


if __name__ == "__main__":
    main()
