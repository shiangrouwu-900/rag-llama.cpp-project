# test_retrieval.py

from rag.embedding import load_embedding_model
from rag.retrieval import load_index, retrieve


def main():
    model = load_embedding_model()
    embeddings, chunks = load_index("storage")

    test_queries = [
        "AORUS MASTER 16 AM6H 的 CPU 是什麼？",
        "這台筆電支援多少記憶體？",
        "What GPU does AORUS MASTER 16 AM6H use?",
        "螢幕規格是什麼？",
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
