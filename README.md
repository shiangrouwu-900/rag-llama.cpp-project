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
選用 `qwen2.5-1.5b.q4_k_m.gguf` 作為生成模型，並使用 `llama.cpp` / `llama-cpp-python` 進行推論。選擇此模型的原因如下：

1. **符合 4GB VRAM 限制**

   `Qwen2.5-1.5B-Instruct` 經 `Q4_K_M` 量化後，模型體積與推論資源需求較低，較適合作為本專案的低資源 baseline。

2. **支援繁體中文與英文混合提問**

   Qwen2.5 系列具備中英文處理能力，可以需要處理中文、英文或中英混合方式提問及回答。

3. **符合指定推論方式**

   任務要求使用 `llama.cpp` 或 `vLLM` 作為推論引擎。本專案使用 GGUF 格式模型，並透過 `llama.cpp` Python binding 載入模型。

---

## 評測結果分析

使用 `qwen2.5-1.5b.q4_k_m.gguf`，設定 `top_k=3`、`n_gpu_layers=20`。評測資料共 21 筆，題型包含 exact、paraphrase、mix、fuzzy、multi、negative 與 out_of_scope。

### 1. 效能表現

| 指標 | 結果 |
|---|---:|
| 評測筆數 | 27 |
| 平均 TTFT | 約 13.06 秒 |
| TTFT 中位數 | 約 13.57 秒 |
| 最低 TTFT | 約 1.38 秒 |
| 最高 TTFT | 約 23.66 秒 |
| 平均 TPS | 約 6.19 tokens/sec |
| TPS 中位數 | 約 6.35 tokens/sec |
| 平均輸出長度 | 約 64.30 tokens |

從結果來看，TPS 大多落在 5 到 7 tokens/sec，可以正常進行串流輸出，但生成速度不算快。平均 TTFT 約 13 秒，首字延遲偏高，會影響互動體驗。可能原因為 prompt/context 偏長。

### 2. 回答品質分析

模型在「明確、單一欄位、短答案」的規格問題上表現較穩定，例如：

- BZH 的 GPU memory。
- BXH 的 GPU 最大顯示功耗。
- 螢幕更新率。
- 記憶體容量、類型、速度與插槽。
- 電池容量與變壓器功率。

這表示目前的 chunking 與 retrieval 對於明確欄位查詢是有效的。當問題能直接對應到單一規格 chunk 時，通常可以抽取正確答案。

但模型在以下情境仍不穩定：

1. **回答可能自我矛盾**

   有些回答先說「目前資料沒有提供」，後面又給出正確規格，表示模型雖然讀到資料，但沒有穩定遵守回答規則。

2. **容易加入不相關內容**

   部分回答在正確答案後，會繼續輸出其他規格或其他問題內容，代表停止條件與 prompt 約束仍不夠穩定。

3. **多欄位比較容易混淆**

   在比較 BZH、BYH、BXH 的 GPU 型號、顯示記憶體與功耗時，模型曾把 GPU 顯示記憶體混成系統記憶體 DDR5 5600MHz，多欄位整合上仍有限制。

4. **否定與範圍外問題處理不佳**

   對於「是否有提供 5G 行動網路規格？」模型錯把 1G LAN 解讀成 5G 行動網路。對於「今天天氣如何？」這類範圍外問題，也無法 rule-based 排除問題。

5. **英文完整問句表現不穩定**

   對於 `Does AORUS MASTER 16 AM6H support Windows 11 Pro?`，模型沒有依規則用英文回答，且錯誤判斷只有 BXH 支援 Windows 11 Pro，表示英文完整句的檢索與生成仍需改善。

### 3. RAG Pipeline 分析

目前 pipeline 的有效設計包含：

- 使用結構化產品資料建立 chunk。
- 將 alias、問題形式與比較詞放入 `search_text`，提升檢索命中率。
- 將乾淨的 `content` 放入 prompt，避免 alias 污染生成內容。
- 使用 `top_k=3` 提供模型上下文。
- 使用 streaming 輸出並記錄 TTFT / TPS。

