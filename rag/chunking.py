# rag/chunking.py

import json
import re
from collections import defaultdict
from pathlib import Path


PRODUCT_DATA_PATH = "data/product_info.json"
COMPARE_TERMS = "比較 差異 不同 一樣 相同 共同 版本 差在哪 哪個 哪款"

FIELD_LABELS = {
    "options": "選項",
    "note": "備註",
    "name": "名稱",
    "model": "型號",
    "type": "類型",
    "count": "數量",
    "standard": "規格",
    "supports": "支援",
    "features": "特色",
    "metadata": "補充資訊",
    "disclaimer": "備註",
    "dimensions": "尺寸",
    "width": "寬度",
    "depth": "深度",
    "height": "高度",
    "weight": "重量",
    "color": "顏色",
    "capacity": "容量",
    "max_capacity": "最大容量",
    "power": "功耗",
    "clock": "時脈",
    "speed": "速度",
    "slots": "插槽",
    "memory": "記憶體",
    "left_side": "左側",
    "right_side": "右側",
    "ai_boost": "AI Boost",
    "boost_clock": "Boost Clock",
    "oc": "OC",
}


def load_product_data(path=PRODUCT_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_product_items(product_data):
    if isinstance(product_data, dict):
        return [product_data]

    if isinstance(product_data, list):
        product_items = [item for item in product_data if isinstance(item, dict)]
        if product_items:
            return product_items

    raise TypeError(
        "product_info.json must be a product object or a list of product objects. "
        f"Got {type(product_data).__name__}."
    )


def humanize_key(key):
    if key in FIELD_LABELS:
        return FIELD_LABELS[key]

    text = re.sub(r"[_\-]+", " ", str(key)).strip()
    return text or str(key)


def normalize_value(value):
    if value is True or value == "true":
        return "是"
    if value is False or value == "false":
        return "否"
    if value == "exists":
        return "有"
    if value is None:
        return "未標示"
    if isinstance(value, str):
        return value.replace(" (approx)", " 約")
    return str(value)


def join_values(values):
    return "、".join(normalize_value(value) for value in values if value is not None)


def make_label(path):
    return " / ".join(humanize_key(part) for part in path if part)


def make_path_key(path):
    return ".".join(str(part) for part in path if part)


def flatten_facts(value, path=()):
    """Flatten arbitrary nested product specs into leaf facts."""
    facts = []

    if isinstance(value, dict):
        for key, item in value.items():
            facts.extend(flatten_facts(item, path + (key,)))
        return facts

    if isinstance(value, list):
        if not value:
            facts.append({
                "path": make_path_key(path),
                "label": make_label(path),
                "value": "未標示",
            })
            return facts

        if all(not isinstance(item, (dict, list)) for item in value):
            facts.append({
                "path": make_path_key(path),
                "label": make_label(path),
                "value": join_values(value),
            })
            return facts

        for index, item in enumerate(value, start=1):
            item_path = path + (str(index),)
            if isinstance(item, (dict, list)):
                facts.extend(flatten_facts(item, item_path))
            else:
                facts.append({
                    "path": make_path_key(item_path),
                    "label": make_label(item_path),
                    "value": normalize_value(item),
                })
        return facts

    facts.append({
        "path": make_path_key(path),
        "label": make_label(path),
        "value": normalize_value(value),
    })
    return facts


def collect_extra_facts(spec):
    facts = []

    if spec.get("features"):
        facts.append({
            "path": "features",
            "label": "特色",
            "value": join_values(spec["features"]),
        })

    if "expandable" in spec:
        facts.append({
            "path": "expandable",
            "label": "可擴充",
            "value": normalize_value(spec["expandable"]),
        })

    metadata = spec.get("metadata")
    if metadata:
        facts.extend(flatten_facts(metadata, ("metadata",)))

    return facts


def spec_identity(spec):
    category = spec["category"]
    zh_name = spec.get("zh_name") or category
    aliases = spec.get("aliases", [])
    return category, zh_name, aliases


def model_text(models):
    return "、".join(model for model in models if model)


def make_chunk_id(chunk_type, category, model=None, field_path=None):
    parts = [chunk_type, category]
    if model:
        parts.append(model)
    if field_path:
        parts.append(field_path)
    return ":".join(parts)


def summarize_facts(facts):
    return "；".join(f"{fact['label']}：{fact['value']}" for fact in facts)


def build_question_patterns(category, zh_name, aliases, models, field_fact=None):
    field_label = field_fact["label"] if field_fact else None
    field_value = field_fact["value"] if field_fact else None
    names = list(dict.fromkeys([zh_name, category, *aliases]))
    if field_label:
        names.append(field_label)

    patterns = []
    for name in names:
        patterns.extend([
            f"{name}規格",
            f"{name}是什麼",
            f"{name}多少",
            f"有沒有{name}",
            f"支援{name}嗎",
            f"{name}差異",
            f"{name}比較",
        ])

    models_joined = model_text(models)
    if models_joined:
        target = field_label or zh_name
        patterns.extend([
            f"{models_joined} 的 {target}",
            f"{models_joined} {target} 有不同嗎",
            f"比較 {models_joined} {target}",
            f"{models_joined} 差在哪",
        ])

    if field_value:
        patterns.append(str(field_value))

    return patterns


def build_search_text(spec, family, models, facts=None, field_fact=None):
    category, zh_name, aliases = spec_identity(spec)
    facts = facts if facts is not None else flatten_facts(spec.get("value", {})) + collect_extra_facts(spec)
    value_terms = []

    for fact in facts:
        value_terms.extend([fact["path"], fact["label"], fact["value"]])

    parts = [
        family,
        model_text(models),
        category,
        zh_name,
        " ".join(aliases),
        " ".join(build_question_patterns(category, zh_name, aliases, models, field_fact)),
        " ".join(value_terms),
        COMPARE_TERMS,
    ]
    return "\n".join(part for part in parts if part)


def build_content(spec, family, models):
    category, zh_name, aliases = spec_identity(spec)
    facts = flatten_facts(spec.get("value", {}))
    extra_facts = collect_extra_facts(spec)
    lines = [
        f"產品：{family}",
        f"適用型號：{model_text(models)}",
        f"規格類別：{zh_name}（{category}）",
    ]

    if aliases:
        lines.append(f"可對應問法：{join_values(aliases)}")

    lines.append("規格內容：")
    lines.extend(f"- {fact['label']}：{fact['value']}" for fact in facts)

    if extra_facts:
        lines.append("補充資訊：")
        lines.extend(f"- {fact['label']}：{fact['value']}" for fact in extra_facts)

    return "\n".join(lines)


def build_spec_chunk(spec, family, all_models):
    category, zh_name, _ = spec_identity(spec)
    models = spec.get("applies_to") or all_models
    facts = flatten_facts(spec.get("value", {})) + collect_extra_facts(spec)
    chunk_type = "shared" if set(models) == set(all_models) else "specific"
    model_for_id = None if chunk_type == "shared" else "_".join(models)
    content = build_content(spec, family, models)

    return {
        "id": make_chunk_id(chunk_type, category, model_for_id),
        "type": chunk_type,
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "aliases": spec.get("aliases", []),
        "models": models,
        "facts": facts,
        "content": content,
        "text": content,
        "search_text": build_search_text(spec, family, models, facts=facts),
        "source": "product_info.json",
    }


def build_fact_chunks(spec, family, all_models):
    category, zh_name, _ = spec_identity(spec)
    models = spec.get("applies_to") or all_models
    facts = flatten_facts(spec.get("value", {})) + collect_extra_facts(spec)
    chunks = []

    for fact in facts:
        content = "\n".join([
            f"產品：{family}",
            f"適用型號：{model_text(models)}",
            f"規格類別：{zh_name}（{category}）",
            f"規格欄位：{fact['label']}（{fact['path']}）",
            f"規格值：{fact['value']}",
        ])
        chunks.append({
            "id": make_chunk_id("fact", category, "_".join(models), fact["path"]),
            "type": "fact",
            "family": family,
            "category": category,
            "zh_name": zh_name,
            "aliases": spec.get("aliases", []),
            "models": models,
            "field_path": fact["path"],
            "field_label": fact["label"],
            "field_value": fact["value"],
            "content": content,
            "text": content,
            "search_text": build_search_text(spec, family, models, facts=facts, field_fact=fact),
            "source": "product_info.json",
        })

    return chunks


def comparable_value(spec):
    facts = flatten_facts(spec.get("value", {})) + collect_extra_facts(spec)
    return tuple((fact["path"], fact["value"]) for fact in facts)


def build_category_comparison_chunk(category, specs, family, all_models):
    first = specs[0]
    _, zh_name, _ = spec_identity(first)
    lines = [
        f"產品：{family}",
        f"比較類別：{zh_name}（{category}）",
        "各型號規格：",
    ]
    search_parts = [family, category, zh_name, model_text(all_models), COMPARE_TERMS]
    values_by_model = {}

    for spec in specs:
        models = spec.get("applies_to") or all_models
        facts = flatten_facts(spec.get("value", {})) + collect_extra_facts(spec)
        summary = summarize_facts(facts)

        for model in models:
            values_by_model[model] = comparable_value(spec)
            lines.append(f"- {model}：{summary}")
        search_parts.append(build_search_text(spec, family, models, facts=facts))

    if len(set(values_by_model.values())) == 1:
        lines.append("比較結論：上述型號在此規格類別相同。")
    else:
        lines.append("比較結論：上述型號在此規格類別有差異，請依各型號列出的規格回答。")

    content = "\n".join(lines)
    return {
        "id": make_chunk_id("comparison", category),
        "type": "comparison",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": sorted(values_by_model),
        "content": content,
        "text": content,
        "search_text": "\n".join(search_parts),
        "source": "product_info.json",
    }


def build_field_comparison_chunks(category, specs, family, all_models):
    _, zh_name, _ = spec_identity(specs[0])
    values_by_field = defaultdict(dict)
    labels_by_field = {}
    aliases = specs[0].get("aliases", [])

    for spec in specs:
        models = spec.get("applies_to") or all_models
        for fact in flatten_facts(spec.get("value", {})) + collect_extra_facts(spec):
            labels_by_field[fact["path"]] = fact["label"]
            for model in models:
                values_by_field[fact["path"]][model] = fact["value"]

    chunks = []
    for field_path, model_values in values_by_field.items():
        if len(model_values) < 2:
            continue

        field_label = labels_by_field[field_path]
        unique_values = set(model_values.values())
        lines = [
            f"產品：{family}",
            f"比較類別：{zh_name}（{category}）",
            f"比較欄位：{field_label}（{field_path}）",
        ]
        lines.extend(f"- {model}：{value}" for model, value in sorted(model_values.items()))
        if len(unique_values) == 1:
            lines.append("比較結論：此欄位在上述型號相同。")
        else:
            lines.append("比較結論：此欄位在上述型號不同。")

        pseudo_spec = {"category": category, "zh_name": zh_name, "aliases": aliases, "value": {}}
        field_fact = {"path": field_path, "label": field_label, "value": " ".join(unique_values)}
        content = "\n".join(lines)
        chunks.append({
            "id": make_chunk_id("comparison_field", category, field_path=field_path),
            "type": "comparison",
            "family": family,
            "category": category,
            "zh_name": zh_name,
            "models": sorted(model_values),
            "field_path": field_path,
            "field_label": field_label,
            "content": content,
            "text": content,
            "search_text": build_search_text(
                pseudo_spec,
                family,
                sorted(model_values),
                facts=[field_fact],
                field_fact=field_fact,
            ),
            "source": "product_info.json",
        })

    return chunks


def build_chunks_for_product(product_data):
    family = product_data["family"]
    all_models = product_data["models"]
    specs = product_data["specs"]
    category_groups = defaultdict(list)
    chunks = []

    for spec in specs:
        category_groups[spec["category"]].append(spec)
        chunks.append(build_spec_chunk(spec, family, all_models))
        chunks.extend(build_fact_chunks(spec, family, all_models))

    for category, grouped_specs in category_groups.items():
        chunks.append(build_category_comparison_chunk(category, grouped_specs, family, all_models))
        chunks.extend(build_field_comparison_chunks(category, grouped_specs, family, all_models))

    return chunks


def build_chunks(product_data):
    chunks = []
    for product_item in normalize_product_items(product_data):
        chunks.extend(build_chunks_for_product(product_item))
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
        print(chunk["content"])


if __name__ == "__main__":
    main()
