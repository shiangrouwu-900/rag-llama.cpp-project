# rag/retrieval.py

import json
import re
import numpy as np


MODEL_PATTERN = re.compile(r"\b(BZH|BYH|BXH)\b", re.IGNORECASE)
COMPARE_TERMS = ("比較", "差異", "不同", "一樣", "共同", "版本", "差在哪")


def load_index(storage_dir="storage"):
    embeddings = np.load(f"{storage_dir}/embeddings.npy")

    with open(f"{storage_dir}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embeddings, chunks


def extract_models(query):
    return {match.upper() for match in MODEL_PATTERN.findall(query)}


def lexical_boost(query, chunk):
    boost = 0.0
    query_lower = query.lower()
    query_models = extract_models(query)
    chunk_models = {model.upper() for model in chunk.get("models", [])}

    if query_models and query_models.issubset(chunk_models):
        boost += 0.12
    elif query_models and query_models & chunk_models:
        boost += 0.06

    category = str(chunk.get("category", "")).lower()
    zh_name = str(chunk.get("zh_name", "")).lower()
    if category and category in query_lower:
        boost += 0.08
    if zh_name and zh_name in query_lower:
        boost += 0.08

    if any(term in query for term in COMPARE_TERMS):
        if chunk.get("type") == "comparison":
            boost += 0.18
        if len(query_models) >= 2 and len(chunk_models) >= len(query_models):
            boost += 0.08

    search_text = str(chunk.get("search_text", "")).lower()
    for term in query_lower.split():
        if len(term) >= 2 and term in search_text:
            boost += 0.01

    return boost


def retrieve(query, model, embeddings, chunks, top_k=3):
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    semantic_scores = embeddings @ query_vec
    scores = semantic_scores.copy()

    for idx, chunk in enumerate(chunks):
        scores[idx] += lexical_boost(query, chunk)

    candidate_count = min(len(chunks), max(top_k * 4, top_k))
    top_indices = np.argsort(scores)[-candidate_count:][::-1]

    results = []
    seen_ids = set()
    for idx in top_indices:
        chunk = chunks[idx]
        chunk_id = chunk.get("id", str(idx))
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        results.append({
            "score": float(scores[idx]),
            "semantic_score": float(semantic_scores[idx]),
            "chunk": chunk,
        })
        if len(results) >= top_k:
            break

    return results
