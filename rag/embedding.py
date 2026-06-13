# rag/embedding.py

import warnings

import numpy as np


class HashingEmbeddingModel:
    """Small CPU-only fallback when PyTorch/SentenceTransformer is unavailable."""

    def __init__(self, n_features=4096):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = self.vectorizer.transform(texts)
        array = vectors.astype(np.float32).toarray()
        if normalize_embeddings:
            array = normalize(array)
        return array if convert_to_numpy else array.tolist()


def load_embedding_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    try:
        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(model_name, local_files_only=True)
        except TypeError:
            return SentenceTransformer(model_name)
        except Exception:
            return SentenceTransformer(model_name)
    except Exception as exc:
        warnings.warn(
            "Falling back to HashingEmbeddingModel because SentenceTransformer "
            f"could not be loaded: {exc}",
            RuntimeWarning,
        )
        return HashingEmbeddingModel()


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


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
