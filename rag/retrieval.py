# rag/retrieval.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_index(storage_dir="storage"): #資料讀回記憶體
    embeddings = np.load(f"{storage_dir}/embeddings.npy")

    with open(f"{storage_dir}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embeddings, chunks


def retrieve(query, model, embeddings, chunks, top_k=3):
    query_vec = model.encode( #query轉成embedding
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_vec #計算query跟embedding的相似度

    #排序，取top_k，反轉
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "chunk": chunks[idx]
        }) #向量結果轉換成可用結果

    return results
