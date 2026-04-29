# rag/embedding.py

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    return SentenceTransformer(model_name)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def build_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts, #文字轉向量(list)
        convert_to_numpy=True, #list轉換成ndarray(快速的且可以節省空間的多維度陣列)
        normalize_embeddings=True #讓每個向量長度等於1
    )
    #embeddings.shape = (chunk數量, 向量維度)，numpy可以直接讀

    return embeddings
    
