# rag/chunking.py

import json
from pathlib import Path
from collections import defaultdict


PRODUCT_DATA_PATH = "data/product_info.json"


def load_product_data(path=PRODUCT_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_value(value, prefix=""):
    """
    將 value 裡的 dict / list / primitive 轉成可讀文字。
    用於保留 key-value 結構，同時讓 embedding 可以吃到自然語意。
    """
    parts = []

    if isinstance(value, dict):
        for k, v in value.items():
            label = f"{prefix}.{k}" if prefix else k
            parts.append(flatten_value(v, label))

    elif isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, dict):
                items.append(flatten_value(item))
            else:
                items.append(str(item))
        parts.append(f"{prefix}: " + "、".join(items) if prefix else "、".join(items))

    else:
        parts.append(f"{prefix}: {value}" if prefix else str(value))

    return "；".join([p for p in parts if p])


def format_metadata(spec):
    """
    處理 metadata / features / expandable 等非必要欄位。
    有就寫進 chunk，沒有就略過。
    """
    parts = []

    features = spec.get("features", [])
    if features:
        parts.append("特色：" + "、".join(features))

    if "expandable" in spec:
        expandable_text = "是" if spec["expandable"] else "否"
        parts.append(f"是否可擴充：{expandable_text}")

    metadata = spec.get("metadata", {})
    disclaimers = metadata.get("disclaimer", [])
    if disclaimers:
        parts.append("備註：" + "、".join(disclaimers))

    return "。".join(parts)


def is_shared_spec(spec, all_models):
    return set(spec.get("applies_to", [])) == set(all_models)


def is_specific_spec(spec):
    return len(spec.get("applies_to", [])) == 1


def make_chunk_id(chunk_type, category, model=None):
    if model:
        return f"{chunk_type}:{category}:{model}"
    return f"{chunk_type}:{category}"


def build_shared_chunk(spec, family, all_models):
    category = spec["category"]
    zh_name = spec.get("zh_name", category)
    applies_to = spec.get("applies_to", [])

    value_text = flatten_value(spec.get("value", {}))
    extra_text = format_metadata(spec)

    text = (
        f"{family} 的 {', '.join(applies_to)} 皆適用於 {zh_name}（{category}）規格。"
        f"{zh_name}（{category}）的規格為：{value_text}。"
    )

    if extra_text:
        text += extra_text + "。"

    return {
        "id": make_chunk_id("shared", category),
        "type": "shared",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": applies_to,
        "text": text,
        "source": "product_info.json",
    }


def build_specific_chunk(spec, family):
    category = spec["category"]
    zh_name = spec.get("zh_name", category)
    model = spec.get("applies_to", [None])[0]

    value_text = flatten_value(spec.get("value", {}))
    extra_text = format_metadata(spec)

    text = (
        f"{family} {model} 的 {zh_name}（{category}）規格如下："
        f"{value_text}。"
    )

    if extra_text:
        text += extra_text + "。"

    return {
        "id": make_chunk_id("specific", category, model),
        "type": "specific",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": [model],
        "text": text,
        "source": "product_info.json",
    }


def build_alias_chunk(spec, family, all_models):
    category = spec["category"]
    zh_name = spec.get("zh_name", category)
    aliases = spec.get("aliases", [])

    question_patterns = [
        f"{zh_name}是什麼？",
        f"{category}是什麼？",
        f"支援哪些{zh_name}？",
        f"有沒有{zh_name}？",
        f"{zh_name}的規格為何？",
        f"{category} spec?",
        f"What is the {category}?",
    ]

    text = (
        f"{family} 的 {zh_name}（{category}）相關詞包含："
        f"{'、'.join(aliases)}。"
        f"常見問句包含：{'、'.join(question_patterns)}。"
        f"這些詞都對應到 {zh_name}（{category}）這個規格分類。"
    )

    return {
        "id": make_chunk_id("alias", category),
        "type": "alias",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": all_models,
        "text": text,
        "source": "product_info.json",
    }


def build_comparison_chunk(category, specs, family):
    """
    同一 category 底下有多個 specific spec 時，產生比較 chunk。
    目前最重要的是 GPU，但此函式可泛用到其他未來變體規格。
    """
    zh_name = specs[0].get("zh_name", category)

    lines = [
        f"{family} 各型號在 {zh_name}（{category}）上的差異如下："
    ]

    for spec in specs:
        model = spec["applies_to"][0]
        value_text = flatten_value(spec.get("value", {}))
        extra_text = format_metadata(spec)

        line = f"{model}：{value_text}"
        if extra_text:
            line += f"。{extra_text}"
        lines.append(line)

    lines.append(
        f"如果問題詢問 {category} 差異、{zh_name} 比較、BZH BYH BXH 差在哪，應優先參考這個比較資訊。"
    )

    text = "。".join(lines) + "。"

    return {
        "id": make_chunk_id("comparison", category),
        "type": "comparison",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": [spec["applies_to"][0] for spec in specs],
        "text": text,
        "source": "product_info.json",
    }


def build_chunks(product_data):
    family = product_data["family"]
    all_models = product_data["models"]
    specs = product_data["specs"]

    chunks = []

    # 1. shared / specific / alias chunks
    for spec in specs:
        if is_shared_spec(spec, all_models):
            chunks.append(build_shared_chunk(spec, family, all_models))

        elif is_specific_spec(spec):
            chunks.append(build_specific_chunk(spec, family))

        else:
            # 預留給未來 applies_to 不是全部、也不是單一型號的情況
            chunks.append(build_shared_chunk(spec, family, all_models))

        chunks.append(build_alias_chunk(spec, family, all_models))

    # 2. comparison chunks: group by category
    category_groups = defaultdict(list)

    for spec in specs:
        if is_specific_spec(spec):
            category_groups[spec["category"]].append(spec)

    for category, grouped_specs in category_groups.items():
        if len(grouped_specs) >= 2:
            chunks.append(build_comparison_chunk(category, grouped_specs, family))

    return chunks


def save_chunks(chunks, path="storage/chunks.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    product_data = load_product_data()
    chunks = build_chunks(product_data)
    save_chunks(chunks)

    print(f"Built {len(chunks)} chunks.")
    for chunk in chunks[:5]:
        print("=" * 80)
        print(chunk["id"])
        print(chunk["text"])


if __name__ == "__main__":
    main()
