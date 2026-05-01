# rag/embedding.py

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def get_embedding_text(chunk):
    # search_text can include aliases and question patterns. content/text stays clean
    # for the LLM prompt.
    return chunk.get("search_text") or chunk.get("content") or chunk["text"]


def build_embeddings(chunks, model):
    texts = [get_embedding_text(chunk) for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings
