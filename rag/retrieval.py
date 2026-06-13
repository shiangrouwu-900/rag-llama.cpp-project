# rag/retrieval.py

import json
import re

import numpy as np


COMPARE_TERMS = ("比較", "差異", "不同", "一樣", "相同", "共同", "版本", "差在哪", "哪個", "哪款")
SPEC_HINT_TERMS = (
    "規格", "型號", "支援", "多少", "容量", "功耗", "記憶體", "顯卡", "顯示", "螢幕",
    "連接埠", "接口", "電池", "變壓器", "網路", "藍牙", "尺寸", "重量", "處理器",
    "gpu", "cpu", "ram", "vram", "display", "storage", "battery", "adapter", "port",
    "windows", "wifi", "bluetooth", "thunderbolt", "hdmi", "usb",
)


def load_index(storage_dir="storage"):
    embeddings = np.load(f"{storage_dir}/embeddings.npy")

    with open(f"{storage_dir}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embeddings, chunks


def token_set(text):
    text = str(text).lower()
    latin_tokens = re.findall(r"[a-z0-9][a-z0-9._+\-]*", text)
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return set(latin_tokens + cjk_terms)


def collect_known_models(chunks):
    models = set()
    for chunk in chunks:
        for model in chunk.get("models", []):
            if model:
                models.add(str(model).upper())
    return models


def extract_models(query, chunks=None):
    query_upper = query.upper()
    known_models = collect_known_models(chunks or [])

    if known_models:
        return {
            model
            for model in known_models
            if re.search(rf"(?<![A-Z0-9]){re.escape(model)}(?![A-Z0-9])", query_upper)
        }

    return set(re.findall(r"(?<![A-Z0-9])[A-Z]{2,5}[0-9A-Z]*(?![A-Z0-9])", query_upper))


def query_has_spec_hint(query, chunks):
    query_lower = query.lower()

    if any(term in query_lower for term in SPEC_HINT_TERMS):
        return True

    if extract_models(query, chunks):
        return True

    for chunk in chunks:
        for key in ("family", "category", "zh_name", "field_label", "field_value"):
            value = str(chunk.get(key, "")).lower()
            if value and value in query_lower:
                return True

        for alias in chunk.get("aliases", []):
            alias = str(alias).lower()
            if alias and alias in query_lower:
                return True

    return False


def field_match_boost(query_lower, chunk):
    boost = 0.0
    for key in ("category", "zh_name", "field_label", "field_path", "field_value"):
        value = str(chunk.get(key, "")).lower()
        if value and value in query_lower:
            boost += 0.20 if key == "category" else 0.08 if key in ("zh_name", "field_label") else 0.04

    for alias in chunk.get("aliases", []):
        alias = str(alias).lower()
        if alias and alias in query_lower:
            boost += 0.04

    category = str(chunk.get("category", "")).lower()
    gpu_memory_query = any(term in query_lower for term in ("gpu memory", "vram", "顯存", "顯示記憶體"))
    if gpu_memory_query and category == "gpu":
        boost += 0.30
    elif gpu_memory_query and category == "memory":
        boost -= 0.20

    return boost


def lexical_boost(query, chunk, chunks=None):
    boost = 0.0
    query_lower = query.lower()
    query_models = extract_models(query, chunks)
    chunk_models = {str(model).upper() for model in chunk.get("models", [])}

    if query_models and query_models.issubset(chunk_models):
        boost += 0.16
    elif query_models and query_models & chunk_models:
        boost += 0.08

    boost += field_match_boost(query_lower, chunk)

    is_compare_query = any(term in query for term in COMPARE_TERMS) or len(query_models) >= 2
    if is_compare_query:
        if chunk.get("type") == "comparison":
            boost += 0.20
        if len(query_models) >= 2 and query_models.issubset(chunk_models):
            boost += 0.08
    elif chunk.get("type") == "fact":
        boost += 0.04

    search_text = str(chunk.get("search_text", "")).lower()
    matched_terms = [term for term in token_set(query_lower) if len(term) >= 2 and term in search_text]
    boost += min(0.12, len(matched_terms) * 0.015)

    return boost


def retrieve(query, model, embeddings, chunks, top_k=4, min_score=0.18, require_spec_hint=True):
    if require_spec_hint and not query_has_spec_hint(query, chunks):
        return []

    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    semantic_scores = embeddings @ query_vec
    scores = semantic_scores.copy()

    for idx, chunk in enumerate(chunks):
        scores[idx] += lexical_boost(query, chunk, chunks)

    candidate_count = min(len(chunks), max(top_k * 5, top_k))
    top_indices = np.argsort(scores)[-candidate_count:][::-1]

    results = []
    seen_ids = set()
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue

        chunk = chunks[idx]
        chunk_id = chunk.get("id", str(idx))
        if chunk_id in seen_ids:
            continue

        seen_ids.add(chunk_id)
        results.append({
            "score": score,
            "semantic_score": float(semantic_scores[idx]),
            "chunk": chunk,
        })
        if len(results) >= top_k:
            break

    return results
