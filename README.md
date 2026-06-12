## 啟動步驟

1. **下載 GitHub 專案**  
   從 GitHub clone 專案到本地。

2. **安裝 uv**  
   安裝 uv 以管理 Python 環境。

3. **執行 `uv sync`**  
   依 `pyproject.toml` 建立環境。

4. **指定模型路徑**  
   透過環境變數 MODEL_PATH 指定要載入的 GGUF 模型檔案路徑  
   量化後的qwen2.5-1.5b.q4_k_m.gguf模型：https://drive.google.com/file/d/1F4CNJoJ4SPFMIg8VZXk4bLuiI7vyhwUw/view?usp=sharing

6. **執行 `main.py`**  
   啟動多輪產品規格問答。

7. **執行 `evaluation.py`**  
   測試 TTFT、TPS 與回答品質。

8. **執行 `vram.py`**  
   量測模型推論 VRAM 使用量。

---

## 檔案說明

### 根目錄

| 檔案 | 說明 |
|---|---|
| `main.py` | 多輪對話主程式。 |
| `evaluation.py` | 以測試集計算 TTFT/TPS。 |
| `vram.py` | 量測 VRAM 使用量。 |
| `vram_samples.csv` | VRAM 使用量紀錄。 |

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
選用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 作為 embedding model。  
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

本次評測以 21 筆測試資料為準，題型包含 exact、paraphrase、mix、fuzzy、multi、negative 與 out_of_scope。

| 指標 | 結果 |
|---|---:|
| 評測筆數 | 21 |
| 平均 TTFT | 約 12.30 秒 |
| TTFT 中位數 | 約 13.23 秒 |
| 最低 TTFT | 約 1.38 秒 |
| 最高 TTFT | 約 23.66 秒 |
| 平均 TPS | 約 6.27 tokens/sec |
| TPS 中位數 | 約 6.36 tokens/sec |
| 最低 TPS | 約 4.60 tokens/sec |
| 最高 TPS | 約 7.48 tokens/sec |
| 平均輸出長度 | 約 63.81 tokens |

從結果來看，TPS 大多落在 5 到 7 tokens/sec，可以正常進行串流輸出，但生成速度不算快。平均 TTFT 約 12.30 秒，首字延遲仍偏高。主要可能原因是 prompt/context 偏長。

### 2. 回答品質分析

模型在「明確、單一欄位、短答案」的規格問題上表現較穩定，例如：

- BZH 的 GPU memory。
- BXH 的 GPU 最大顯示功耗。
- 螢幕更新率。
- 記憶體容量、類型、速度與插槽。
- 電池容量與變壓器功率。

當問題能直接對應到單一規格 chunk 時，模型通常可以抽取正確答案。

但模型在以下情境仍不穩定：

1. **回答可能混入不相關資訊**

   例如詢問 BZH、BYH、BXH 共同 CPU 時，模型有抓到正確 CPU，但同時混入 BXH 的 GPU 資訊。這表示檢索內容或 prompt 中的其他 chunk 可能干擾生成結果。

2. **容易加入多餘內容**

   部分回答在正確答案後，會繼續輸出其他規格或重複使用者問題。例如詢問機身大小與重量時，模型先回答正確尺寸與重量，但後面又接續出現與儲存裝置、顯示晶片相關的句子。這代表停止條件與 prompt 約束仍不夠穩定。

3. **多欄位比較容易混淆**

   在比較 BZH、BYH、BXH 的 GPU 型號、顯示記憶體與功耗時，模型能部分回答 GPU 型號與功耗，但把 GPU 顯示記憶體混成系統記憶體 DDR5 5600MHz。這表示在多欄位整合與跨版本比較上仍有限制。

4. **否定與範圍外問題處理不佳**

   對於「是否有提供 5G 行動網路規格？」模型錯把 1G LAN 解讀成 5G 行動網路。對於「今天天氣如何？」這類範圍外問題，也無法 rule-based 排除問題。

5. **英文完整問句表現不穩定**

   對於 `Does AORUS MASTER 16 AM6H support Windows 11 Pro?`，模型沒有依規則用英文回答，且錯誤判斷只有 BXH 支援 Windows 11 Pro。這表示英文完整句的檢索與生成仍需改善。

### 3. RAG Pipeline 分析

目前 pipeline 的有效設計包含：

- 使用結構化產品資料建立 chunk。
- 將 alias、問題形式與比較詞放入 `search_text`，提升檢索命中率。
- 將乾淨的 `content` 放入 prompt，避免 alias 污染生成內容。
- 使用 `top_k=3` 提供模型上下文。
- 使用 streaming 輸出並記錄 TTFT / TPS。

---

### 4GB VRAM 使用量驗證

為確認本專案符合 4GB VRAM 限制，使用 Colab T4 GPU 測量，包含載入 index、載入 embedding model、載入 LLM、retrieval 與 generation。  
在本次 Colab T4 測試環境與目前設定下，峰值 VRAM 使用量低於 4GB。

測試結果如下：

```text
=== VRAM report ===
Baseline: 0 MB
baseline             peak: 0 MB / 15,360 MB (+0 MB)
idle                 peak: 3 MB / 15,360 MB (+3 MB)
load_index           peak: 3 MB / 15,360 MB (+3 MB)
load_embedding       peak: 473 MB / 15,360 MB (+473 MB)
load_llm             peak: 1,255 MB / 15,360 MB (+1,255 MB)
retrieve             peak: 1,641 MB / 15,360 MB (+1,641 MB)
generate             peak: 1,659 MB / 15,360 MB (+1,659 MB)
finished             peak: 1,659 MB / 15,360 MB (+1,659 MB)
Overall peak:        1,659 MB / 15,360 MB at stage 'generate'
4096 MB check:       PASS (peak increase from baseline: 1,659 MB, limit: 4,096 MB)
