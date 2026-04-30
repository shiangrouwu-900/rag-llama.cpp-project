# main.py
import os
import json
import numpy as np
from pathlib import Path

from rag.chunking import load_product_data, build_chunks
from rag.embedding import load_embedding_model, build_embeddings
from rag.retrieval import retrieve
from rag.generation import load_llm, build_prompt, generate_stream

PRODUCT_DATA_PATH = "data/product_info.json"


def save_artifacts(chunks, embeddings, output_dir="storage"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(f"{output_dir}/embeddings.npy", embeddings)


def main():
    embedding_model = load_embedding_model()

    product_data = load_product_data(PRODUCT_DATA_PATH)
    chunks = build_chunks(product_data)
    embeddings = build_embeddings(chunks, embedding_model)
    save_artifacts(chunks, embeddings, "storage")

    model_path = os.getenv("MODEL_PATH")
    llm = load_llm(model_path)

    print("AI 助手已啟動。")
    print("你可以詢問 GIGABYTE AORUS MASTER 16 AM6H 產品規格。")
    print("輸入 exit / quit / q 離開。")

    while True:
        query = input("\n請輸入問題：").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("結束程式。")
            break

        if not query:
            print("請輸入有效問題。")
            continue

        results = retrieve(
            query=query,
            model=embedding_model,
            embeddings=embeddings,
            chunks=chunks,
            top_k=3,
        )

        prompt = build_prompt(query, results)

        print("\n回答：")
        answer, _ = generate_stream(llm, prompt)
        print()


if __name__ == "__main__":
    main()
