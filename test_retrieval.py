from rag.chunking import load_product_data, build_chunks
from rag.embedding import load_embedding_model, build_embeddings
from rag.retrieval import retrieve


def main():
    print("載入產品資料...")
    product_data = load_product_data("data/product_info.json")

    print("建立 chunks...")
    chunks = build_chunks(product_data)
    print(f"Chunks 數量: {len(chunks)}")

    print("載入 embedding model...")
    model = load_embedding_model()

    print("現場建立 embeddings...")
    embeddings = build_embeddings(chunks, model)
    print(f"Embeddings shape: {embeddings.shape}")

    test_queries = [
        "AORUS MASTER 16 BXH 的 GPU 最大顯示功耗是多少？",
        "BZH、BYH、BXH 這三個版本的處理器有不同嗎？",
        "Does AORUS MASTER 16 AM6H support Windows 11 Pro？",
        "AORUS MASTER 16 AM6H 的無線網路、藍牙、網路孔與視訊鏡頭規格是什麼？",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"Query: {query}")

        results = retrieve(
            query=query,
            model=model,
            embeddings=embeddings,
            chunks=chunks,
            top_k=3,
        )

        for i, result in enumerate(results, start=1):
            chunk = result["chunk"]

            print(f"\nTop {i}")
            print(f"Score: {result['score']:.4f}")
            print(f"Chunk ID: {chunk.get('id')}")
            print(f"Type: {chunk.get('type')}")
            print(f"Category: {chunk.get('category')}")
            print(f"Models: {chunk.get('models')}")
            print("Text:")
            print(chunk["text"][:800])


if __name__ == "__main__":
    main()
