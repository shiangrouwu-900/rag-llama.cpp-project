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
