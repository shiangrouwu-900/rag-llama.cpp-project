def load_embedding_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embedding_text(chunk):
    return chunk.get("search_text") or chunk.get("content") or chunk["text"]


def build_embeddings(chunks, model):
    texts = [get_embedding_text(chunk) for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings
