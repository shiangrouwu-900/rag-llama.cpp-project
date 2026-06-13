# rag/evaluation.py
import json
import os
from datetime import datetime
from pathlib import Path

from rag.build_index import main as build_index_main
from rag.embedding import load_embedding_model
from rag.generation import build_prompt, generate_stream, load_llm
from rag.retrieval import load_index, retrieve


TOP_K = 4
MODEL_NAME = "Qwen3-1.7B-Q4_K_M"
MODEL_PATH = os.getenv("MODEL_PATH", "models/Qwen3-1.7B-Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))

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


def normalize_text(text):
    return str(text or "").lower().replace(" ", "")


def contains_all(text, facts):
    normalized = normalize_text(text)
    return all(normalize_text(fact) in normalized for fact in facts)


def contains_any(text, facts):
    normalized = normalize_text(text)
    return any(normalize_text(fact) in normalized for fact in facts)


def summarize_retrieval(results):
    summary = []
    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        summary.append({
            "rank": rank,
            "id": chunk.get("id"),
            "type": chunk.get("type"),
            "category": chunk.get("category"),
            "zh_name": chunk.get("zh_name"),
            "models": chunk.get("models", []),
            "field_path": chunk.get("field_path"),
            "field_label": chunk.get("field_label"),
            "score": result.get("score"),
            "semantic_score": result.get("semantic_score"),
        })
    return summary


def evaluate_retrieval(results, expected_chunks):
    if not expected_chunks:
        return None

    retrieved_ids = {result["chunk"].get("id") for result in results}
    expected_ids = set(expected_chunks)
    return {
        "expected_chunks": list(expected_ids),
        "retrieved_expected_chunks": sorted(expected_ids & retrieved_ids),
        "hit": bool(expected_ids & retrieved_ids),
    }


def evaluate_answer(answer, required_facts=None, forbidden_facts=None):
    required_facts = required_facts or []
    forbidden_facts = forbidden_facts or []

    return {
        "required_facts": required_facts,
        "forbidden_facts": forbidden_facts,
        "required_facts_pass": contains_all(answer, required_facts) if required_facts else None,
        "forbidden_facts_pass": not contains_any(answer, forbidden_facts) if forbidden_facts else None,
    }


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
            "export MODEL_PATH=/content/drive/MyDrive/rag_models/Qwen3-1.7B-Q4_K_M.gguf"
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

        retrieval_eval = evaluate_retrieval(results, item.get("expected_chunks", []))
        answer_eval = evaluate_answer(
            answer,
            required_facts=item.get("required_facts", []),
            forbidden_facts=item.get("forbidden_facts", []),
        )

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "expected_answer": item.get("expected_answer"),
            "type": item.get("type"),
            "answer": answer.strip(),
            "retrieved_chunks": summarize_retrieval(results),
            "retrieval_eval": retrieval_eval,
            "answer_eval": answer_eval,
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
