# rag/chunking.py

import json
from pathlib import Path
from collections import defaultdict


PRODUCT_DATA_PATH = "data/product_info.json"

CATEGORY_LABELS = {
    "OS": "作業系統",
    "CPU": "處理器",
    "GPU": "顯示晶片",
    "Display": "螢幕",
    "Memory": "記憶體",
    "Storage": "儲存裝置",
    "Keyboard": "鍵盤",
    "Ports": "連接埠",
    "Audio": "音效",
    "Connectivity": "通訊",
    "Webcam": "視訊鏡頭",
    "Security": "安全裝置",
    "Battery": "電池",
    "Adapter": "變壓器",
    "Physical": "外觀與尺寸",
}

FIELD_LABELS = {
    "options": "選項",
    "note": "備註",
    "name": "型號",
    "cache": "快取",
    "clock": "時脈",
    "cores": "核心",
    "threads": "執行緒",
    "model": "型號",
    "memory": "記憶體",
    "power": "功耗",
    "ai_boost": "AI Boost",
    "boost_clock": "Boost Clock",
    "oc": "OC",
    "size": "尺寸",
    "aspect_ratio": "比例",
    "panel": "面板",
    "resolution_name": "解析度等級",
    "resolution": "解析度",
    "refresh_rate": "更新率",
    "response_time": "反應時間",
    "color_gamut": "色域",
    "brightness": "亮度",
    "contrast_ratio": "對比",
    "max_capacity": "最大容量",
    "type": "類型",
    "speed": "速度",
    "slots": "插槽",
    "backlight": "背光",
    "zones": "分區",
    "key_travel": "鍵程",
    "left_side": "左側",
    "right_side": "右側",
    "count": "數量",
    "port": "連接埠",
    "standard": "規格",
    "supports": "支援",
    "speakers": "喇叭",
    "power_each": "單顆功率",
    "microphone": "麥克風",
    "wifi": "Wi-Fi",
    "protocol": "協定",
    "antenna": "天線",
    "lan": "有線網路",
    "bluetooth": "藍牙",
    "camera": "鏡頭",
    "tpm": "TPM",
    "capacity": "容量",
    "dimensions": "尺寸",
    "width": "寬",
    "depth": "深",
    "height": "高",
    "weight": "重量",
    "color": "顏色",
}


def load_product_data(path=PRODUCT_DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_for(key):
    return FIELD_LABELS.get(key, key)


def list_text(items):
    return "、".join(str(item) for item in items)


def normalize_value(value):
    if value == "exists":
        return "有"
    if value == "true":
        return "是"
    if value == "false":
        return "否"
    if isinstance(value, str):
        return value.replace(" (approx)", " 約")
    return str(value)


def format_port_item(item):
    count = item.get("count", 1)
    port = item.get("port", "")
    standard = item.get("standard")
    supports = item.get("supports", [])

    text = f"{count} 個 {port}" if count else port
    if standard:
        text += f"（{standard}）"
    if supports:
        text += f"，支援 {list_text(supports)}"
    return text


def flatten_value(value, prefix=""):
    """Return a readable fallback for categories without custom formatting."""
    parts = []

    if isinstance(value, dict):
        for key, item in value.items():
            label = label_for(key)
            if isinstance(item, dict):
                nested = flatten_value(item)
                parts.append(f"{label}為 {nested}")
            elif isinstance(item, list):
                if item and all(isinstance(x, dict) for x in item):
                    parts.append(f"{label}包含 " + "；".join(format_port_item(x) for x in item))
                else:
                    parts.append(f"{label}為 {list_text(item)}")
            else:
                parts.append(f"{label}為 {normalize_value(item)}")
    elif isinstance(value, list):
        parts.append(list_text(value))
    else:
        parts.append(normalize_value(value))

    return "，".join(part for part in parts if part)


def format_value(category, value):
    """Format structured specs as natural answer-ready Chinese."""
    if category == "Physical":
        dimensions = value.get("dimensions", {})
        size = " x ".join(
            dimensions[key]
            for key in ("width", "depth", "height")
            if dimensions.get(key)
        )
        parts = []
        if size:
            parts.append(f"尺寸為 {size}")
        if value.get("weight"):
            parts.append(f"重量約 {value['weight'].replace(' (approx)', '')}")
        if value.get("color"):
            parts.append(f"顏色為 {value['color']}")
        return "，".join(parts) + "。"

    if category == "CPU":
        parts = []
        if value.get("name"):
            parts.append(value["name"])
        details = []
        if value.get("cores"):
            details.append(f"{value['cores']} 核心")
        if value.get("threads"):
            details.append(f"{value['threads']} 執行緒")
        if value.get("clock"):
            details.append(f"最高 {value['clock'].replace('up to ', '')}")
        if value.get("cache"):
            details.append(f"{value['cache']} 快取")
        return "處理器為 " + "，".join(parts + details) + "。"

    if category == "GPU":
        clock = value.get("clock", {})
        details = []
        if value.get("model"):
            details.append(value["model"])
        if value.get("memory"):
            details.append(value["memory"])
        if value.get("power"):
            details.append(value["power"])
        if clock.get("ai_boost"):
            details.append(f"AI Boost {clock['ai_boost']}")
        if clock.get("boost_clock"):
            details.append(f"Boost Clock {clock['boost_clock']}")
        return "顯示晶片為 " + "，".join(details) + "。"

    if category == "Display":
        parts = [
            value.get("size"),
            value.get("resolution_name"),
            value.get("resolution"),
            value.get("panel"),
            value.get("refresh_rate"),
            value.get("response_time"),
            value.get("color_gamut"),
            value.get("brightness"),
        ]
        return "螢幕規格為 " + "，".join(part for part in parts if part) + "。"

    if category == "Memory":
        return (
            f"記憶體為 {value.get('type')} {value.get('speed')}，"
            f"{value.get('slots')}，最高支援 {value.get('max_capacity')}。"
        )

    if category == "Storage":
        return (
            f"儲存裝置為 {value.get('type')}，最高 {value.get('max_capacity')}，"
            f"插槽包含 {list_text(value.get('slots', []))}。"
        )

    if category == "Ports":
        left = "；".join(format_port_item(item) for item in value.get("left_side", []))
        right = "；".join(format_port_item(item) for item in value.get("right_side", []))
        return f"連接埠左側包含 {left}；右側包含 {right}。"

    return flatten_value(value) + "。"


def format_metadata(spec):
    parts = []

    features = spec.get("features", [])
    if features:
        parts.append("特色：" + list_text(features))

    if "expandable" in spec:
        parts.append("可擴充：" + ("是" if spec["expandable"] else "否"))

    disclaimers = spec.get("metadata", {}).get("disclaimer", [])
    if disclaimers:
        parts.append("備註：" + list_text(disclaimers))

    return " ".join(parts)


def is_shared_spec(spec, all_models):
    return set(spec.get("applies_to", [])) == set(all_models)


def is_specific_spec(spec):
    return len(spec.get("applies_to", [])) == 1


def make_chunk_id(chunk_type, category, model=None):
    if model:
        return f"{chunk_type}:{category}:{model}"
    return f"{chunk_type}:{category}"


def build_question_patterns(category, zh_name, aliases, models):
    model_text = "、".join(models)
    names = [zh_name, category] + aliases
    patterns = []
    for name in names:
        patterns.extend([
            f"{name}規格為何",
            f"{name}是什麼",
            f"有沒有{name}",
            f"支援{name}嗎",
            f"{name}多少",
        ])
    patterns.extend([
        f"{model_text} 的 {zh_name} 有不同嗎",
        f"{model_text} 有什麼差異",
        f"{model_text} 差在哪",
        f"比較 {model_text}",
    ])
    return patterns


def build_search_text(spec, family, models, include_compare_terms=True):
    category = spec["category"]
    zh_name = spec.get("zh_name") or CATEGORY_LABELS.get(category, category)
    aliases = spec.get("aliases", [])
    value_text = flatten_value(spec.get("value", {}))
    question_patterns = build_question_patterns(category, zh_name, aliases, models)

    parts = [
        family,
        " ".join(models),
        category,
        zh_name,
        " ".join(aliases),
        " ".join(question_patterns),
        value_text,
    ]
    if include_compare_terms:
        parts.append("比較 差異 不同 一樣 共同 版本")
    return "\n".join(part for part in parts if part)


def build_content(spec, family, models):
    category = spec["category"]
    zh_name = spec.get("zh_name") or CATEGORY_LABELS.get(category, category)
    model_text = "、".join(models)
    value_text = format_value(category, spec.get("value", {}))
    extra_text = format_metadata(spec)

    content = f"{family} {model_text} 的{zh_name}：{value_text}"
    if extra_text:
        content += f"\n{extra_text}"
    return content


def build_shared_chunk(spec, family, all_models):
    category = spec["category"]
    zh_name = spec.get("zh_name") or CATEGORY_LABELS.get(category, category)
    models = spec.get("applies_to", [])
    content = build_content(spec, family, models)

    return {
        "id": make_chunk_id("shared", category),
        "type": "shared",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": models,
        "content": content,
        "text": content,
        "search_text": build_search_text(spec, family, models),
        "source": "product_info.json",
    }


def build_specific_chunk(spec, family):
    category = spec["category"]
    zh_name = spec.get("zh_name") or CATEGORY_LABELS.get(category, category)
    model = spec.get("applies_to", [None])[0]
    content = build_content(spec, family, [model])

    return {
        "id": make_chunk_id("specific", category, model),
        "type": "specific",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": [model],
        "content": content,
        "text": content,
        "search_text": build_search_text(spec, family, [model]),
        "source": "product_info.json",
    }


def build_comparison_chunk(category, specs, family):
    zh_name = specs[0].get("zh_name") or CATEGORY_LABELS.get(category, category)
    models = [spec["applies_to"][0] for spec in specs]
    contents = []
    search_parts = [family, category, zh_name, "比較 差異 不同 一樣 共同 版本", " ".join(models)]

    values_by_model = {}
    for spec in specs:
        model = spec["applies_to"][0]
        value_text = format_value(category, spec.get("value", {})).rstrip("。")
        values_by_model[model] = value_text
        contents.append(f"{model}：{value_text}。")
        search_parts.append(build_search_text(spec, family, [model]))

    unique_values = set(values_by_model.values())
    if len(unique_values) == 1:
        summary = f"{'、'.join(models)} 的{zh_name}沒有不同，皆為 {next(iter(unique_values))}。"
    else:
        summary = f"{'、'.join(models)} 的{zh_name}有差異：" + " ".join(contents)

    content = f"{family} 版本比較（{zh_name}）：{summary}"

    return {
        "id": make_chunk_id("comparison", category),
        "type": "comparison",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": models,
        "content": content,
        "text": content,
        "search_text": "\n".join(search_parts),
        "source": "product_info.json",
    }


def build_shared_comparison_chunk(category, specs, family, all_models):
    spec = specs[0]
    zh_name = spec.get("zh_name") or CATEGORY_LABELS.get(category, category)
    value_text = format_value(category, spec.get("value", {})).rstrip("。")
    model_text = "、".join(all_models)
    content = f"{family} 版本比較（{zh_name}）：{model_text} 的{zh_name}沒有不同，皆為 {value_text}。"

    search_text = "\n".join([
        build_search_text(spec, family, all_models),
        f"{model_text} {zh_name} 是否不同 有不同嗎 一樣嗎 共同 差異 比較",
    ])

    return {
        "id": make_chunk_id("comparison_shared", category),
        "type": "comparison",
        "family": family,
        "category": category,
        "zh_name": zh_name,
        "models": all_models,
        "content": content,
        "text": content,
        "search_text": search_text,
        "source": "product_info.json",
    }


def build_chunks(product_data):
    family = product_data["family"]
    all_models = product_data["models"]
    specs = product_data["specs"]

    chunks = []
    category_groups = defaultdict(list)

    for spec in specs:
        category_groups[spec["category"]].append(spec)

        if is_shared_spec(spec, all_models):
            chunks.append(build_shared_chunk(spec, family, all_models))
        elif is_specific_spec(spec):
            chunks.append(build_specific_chunk(spec, family))
        else:
            chunks.append(build_shared_chunk(spec, family, all_models))

    for category, grouped_specs in category_groups.items():
        specific_specs = [spec for spec in grouped_specs if is_specific_spec(spec)]
        if len(specific_specs) >= 2:
            chunks.append(build_comparison_chunk(category, specific_specs, family))
        elif len(grouped_specs) == 1 and is_shared_spec(grouped_specs[0], all_models):
            chunks.append(build_shared_comparison_chunk(category, grouped_specs, family, all_models))

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
