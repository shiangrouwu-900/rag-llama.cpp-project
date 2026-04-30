# test_retrieval.py

from rag.embedding import load_embedding_model
from rag.retrieval import load_index, retrieve


def main():
    model = load_embedding_model()
    embeddings, chunks = load_index("storage")

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
            print(f"Chunk ID: {chunk.get('chunk_id')}")
            print(f"Category: {chunk.get('metadata', {}).get('category')}")
            print("Text:")
            print(chunk["text"][:800])


if __name__ == "__main__":
    main()
