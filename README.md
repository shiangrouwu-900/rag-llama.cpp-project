## 啟動步驟

1. **下載 GitHub 專案**  
   從 GitHub clone 專案到本地。

2. **安裝 uv**  
   安裝 uv 以管理 Python 環境。

3. **執行 `uv sync`**  
   依 `pyproject.toml` 建立環境。

4. **指定模型雲端路徑**  
   量化後 GGUF 模型位置位於 model/ 中。

5. **執行 `main.py`**  
   啟動多輪產品規格問答。

6. **執行 `evaluation.py`**  
   測試 TTFT、TPS 與回答品質。

7. **執行 `vram.py`**  
   量測模型推論 VRAM 使用量。

---

## 檔案說明

### 根目錄

| 檔案 | 說明 |
|---|---|
| `main.py` | 多輪對話主程式。 |
| `evaluation.py` | 以測試集計算 TTFT/TPS。 |
| `vram.py` | 量測 VRAM 使用量。 |

### `rag/` 資料夾

| 檔案 | 說明 |
|---|---|
| `build_index.py` | 建立 chunks 與 embeddings。 |
| `chunking.py` | 將規格資料切成 chunks。 |
| `embedding.py` | 載入模型並產生向量。 |
| `retrieval.py` | 檢索最相關產品資料。 |
| `generation.py` | 建立 prompt 並串流生成。 |

### `data/` 資料夾

| 檔案 | 說明 |
|---|---|
| `product_info.json` | AORUS 規格結構化資料。 |
| `test_data.json` | 評測用問題與標準答案。 |

### `storage/` 資料夾

| 檔案 | 說明 |
|---|---|
| `chunks.json` | 已切分的規格 chunks。 |
| `embeddings.npy` | chunks 對應的向量資料。 |

---

## 模型選擇理由

---

## 評測結果分析



