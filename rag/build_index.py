# build_index.py
import json
import numpy as np
from pathlib import Path

from rag.chunking import load_product_data, build_chunks
from rag.embedding import load_embedding_model, build_embeddings


PRODUCT_DATA_PATH = "data/product_info.json"
OUTPUT_DIR = "storage"


def save_index(chunks, embeddings, output_dir=OUTPUT_DIR):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(f"{output_dir}/embeddings.npy", embeddings)


def main():
    print("載入產品資料...")
    product_data = load_product_data(PRODUCT_DATA_PATH)

    print("建立 chunks...")
    chunks = build_chunks(product_data)
    print(f"Chunks 數量: {len(chunks)}")

    print("載入 embedding model...")
    embedding_model = load_embedding_model()

    print("建立 embeddings...")
    embeddings = build_embeddings(chunks, embedding_model)
    print(f"Embeddings shape: {embeddings.shape}")

    print("儲存 index...")
    save_index(chunks, embeddings, OUTPUT_DIR)

    print("完成。")


if __name__ == "__main__":
    main()
