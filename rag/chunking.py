# rag/chunking.py

import json
from pathlib import Path


def flatten_value(value, prefix=""):
    lines = []
    #處理dict
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix} {k}".strip().replace("_", " ").title()
            lines.extend(flatten_value(v, key))
    #處理list
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                parts = []
                for k, v in item.items():
                    if isinstance(v, list):
                        parts.append(f"{k}: {', '.join(map(str, v))}")
                    else:
                        parts.append(f"{k}: {v}")
                lines.append(f"{prefix}: " + ", ".join(parts))
            else:
                lines.append(f"{prefix}: {item}")

    else:
        lines.append(f"{prefix}: {value}")

    return lines


def build_chunks(json_path: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    chunks = []

    for item in data:
        models = item.get("models") or [item.get("model")]
        category = item["category"]

        lines = [
            f"Models: {', '.join(models)}",
            f"Category: {category}"
        ]

        lines.extend(flatten_value(item.get("content", {})))

        if item.get("features"):
            lines.append("Features: " + ", ".join(item["features"]))

        text = "\n".join(lines)

        chunks.append({
            "chunk_id": f"{'_'.join(models)}_{category}",
            "text": text,
            "metadata": {
                "models": models,
                "category": category
            }
        })

    return chunks
