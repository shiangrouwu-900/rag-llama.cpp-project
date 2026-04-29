# rag/evaluation.py
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from rag.chunking import build_chunks
from rag.embedding import load_embedding_model, build_embeddings
from rag.retrieval import retrieve
from rag.generation import load_llm, build_prompt, generate_stream


TOP_K = 3
MODEL_NAME = "qwen2.5-1.5b.q4_k_m"
MODEL_PATH = "models/qwen2.5-1.5b.q4_k_m.gguf"
N_GPU_LAYERS = 20

TEST_DATA_PATH = "data/test_data.json"
PRODUCT_DATA_PATH = "data/product_info.json"
OUTPUT_PATH = "storage/evaluation_records.jsonl"


def load_test_data(path=TEST_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(record, path=OUTPUT_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("載入 embedding model...")
    embedding_model = load_embedding_model()

    print("建立 chunks / embeddings...")
    chunks = build_chunks(PRODUCT_DATA_PATH)
    embeddings = build_embeddings(chunks, embedding_model)

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
            "n_gpu_layers": N_GPU_LAYERS,
        }

        save_jsonl(record)

        print("metrics:")
        print(json.dumps(record, ensure_ascii=False, indent=2))

    print("\nEvaluation 完成。")


if __name__ == "__main__":
    main()