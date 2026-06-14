# RAG llama.cpp Product QA Project

本專案是一個以 `llama.cpp` / `llama-cpp-python` 執行 GGUF 語言模型的產品規格問答系統。流程包含產品資料 chunking、embedding index 建立、hybrid retrieval、prompt generation、evaluation，以及 VRAM 使用量測試。

---

## 啟動流程

1. **Clone GitHub 專案**

   ```bash
   git clone https://github.com/shiangrouwu-900/rag-llama.cpp-project.git
   cd rag-llama.cpp-project
   ```

2. **安裝 uv 環境**

   ```bash
   uv sync
   ```

3. **下載語言模型**

   目前程式預設使用：

   ```text
   Qwen3-1.7B-Q4_K_M.gguf
   ```

   可從 Hugging Face 下載：

   ```text
   unsloth/Qwen3-1.7B-GGUF
   ```

   建議放置於：

   ```text
   models/Qwen3-1.7B-Q4_K_M.gguf
   ```

   若模型放在其他位置，請設定 `MODEL_PATH`：

   ```bash
   MODEL_PATH=/path/to/Qwen3-1.7B-Q4_K_M.gguf uv run python main.py
   ```

4. **執行主程式**

   ```bash
   uv run python main.py
   ```

5. **執行 evaluation**

   ```bash
   uv run python evaluation.py
   ```

6. **執行 VRAM test**

   ```bash
   uv run python vram_test.py
   ```

---

## 檔案說明

### 主要程式

| `main.py` | 互動式產品規格問答入口，負責載入 index、embedding model、LLM，並串接 retrieval 與 generation。 |
| `evaluation.py` | 使用 `data/test_data.json` 執行測試問題，記錄回答、retrieved chunks、推論速度與模型參數。 |
| `vram_test.py` | 量測 RAG pipeline 各階段 GPU VRAM 使用量，包含載入 index、embedding model、LLM、retrieval 與 generation。 |
| `COLAB_T4_GPU_SETUP.md` | Colab T4 GPU 執行流程、CUDA 版 `llama-cpp-python` 安裝方式與測試指令。 |
| `configs/colab_t4.env` | Colab T4 GPU 的執行參數設定，例如 `N_GPU_LAYERS`、`N_CTX`、`TOP_K`、`MAX_TOKENS`。 |
| `vram_samples.csv` | VRAM test 的取樣結果。 |

### `rag/`

| 檔案 | 作用 |
|---|---|
| `build_index.py` | 從產品 JSON 建立 chunks 與 embeddings。 |
| `chunking.py` | 將產品規格 JSON 轉換為 shared、specific、fact、comparison 等多層 chunk。 |
| `embedding.py` | 載入 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，並將 chunk 文字轉為 embedding。 |
| `retrieval.py` | 執行 hybrid retrieval，結合 embedding similarity、型號提示、category hint、field hint 與 lexical boost。 |
| `generation.py` | 建立 Qwen3 Instruct prompt，載入 `llama-cpp-python`，並產生最終回答。 |

### `data/`

| 檔案 | 作用 |
|---|---|
| `product_info.json` | 產品規格來源資料。 |
| `test_data.json` | Evaluation 測試問題、期望答案與題型。 |

### `storage/`

| 檔案 | 作用 |
|---|---|
| `chunks.json` | 由 `rag/chunking.py` 產生的 chunks。 |
| `embeddings.npy` | 對應 `chunks.json` 的 embedding matrix。 |
| `evaluation_records.jsonl` | Evaluation 執行結果。 |

### GitHub Actions

| 檔案 | 作用 |
|---|---|
| `.github/workflows/uv-smoke-test.yml` | 在 GitHub Actions 中執行 `uv sync --locked`、Python 編譯檢查、核心模組匯入與 index build。 |

---

## 使用的模型與選擇理由

### 語言模型

目前使用：

```text
Qwen3-1.7B-GGUF / Qwen3-1.7B-Q4_K_M.gguf
```

選擇理由：

1. **適合 GGUF 與 llama.cpp 生態**

   專案以 `llama-cpp-python` 作為推論後端，GGUF 模型可以直接由 llama.cpp 載入，並支援 `n_gpu_layers` 進行 GPU offload。

2. **推論成本低**

   Qwen3-1.7B 的 Q4_K_M 量化版本模型檔較小，能降低 VRAM 使用量與推論延遲，適合產品規格 QA 這類短回答任務。

3. **中英文混合能力足夠**

   測試資料中同時包含中文問句、英文規格名稱、產品型號與技術詞彙，例如 `GPU memory`、`Windows 11 Pro`、`Thunderbolt`、`DisplayPort`。Qwen3 Instruct 對這類混合輸入有足夠的理解能力。

4. **可搭配 `/no_think` 控制輸出**

   `generation.py` 的 prompt 使用 `/no_think`，並要求模型依據 retrieved context 直接回答規格內容，減少推理痕跡與額外輸出。

### Embedding 模型

目前使用：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

選擇理由：

1. 支援多語 retrieval，適合中文問題與英文規格欄位混合的資料。
2. 向量維度較小，建立與查詢 index 的成本低。
3. 與目前 `storage/embeddings.npy` 的 index 格式一致。

---

## 評測結果分析

以下分析依據 GitHub 目前的 `storage/evaluation_records.jsonl`，共 21 筆測試資料，包含 8 筆 exact、4 筆 paraphrase、3 筆 mix、2 筆 fuzzy、2 筆 multi、1 筆 negative，以及 1 筆 out_of_scope。

測試參數：

| 參數 | 數值 |
|---|---:|
| model | `Qwen3-1.7B-Q4_K_M` |
| top_k | 4 |
| n_gpu_layers | -1 |
| n_ctx | 3072 |
| max_tokens | 192 |

### 1. 效能表現

| 指標 | 結果 |
|---|---:|
| 平均 TTFT | 0.119 秒 |
| TTFT 中位數 | 0.092 秒 |
| 最低 TTFT | 0.029 秒 |
| 最高 TTFT | 0.385 秒 |
| 平均 TPS | 99.07 tokens/sec |
| TPS 中位數 | 99.77 tokens/sec |
| 最低 TPS | 89.85 tokens/sec |
| 最高 TPS | 108.25 tokens/sec |
| 平均輸出長度 | 31.38 tokens |

整體速度表現穩定。`n_gpu_layers=-1` 時，Qwen3-1.7B Q4_K_M 的輸出速度約落在 90 到 108 tokens/sec，短規格回答的 TTFT 多數低於 0.15 秒。較高的 TTFT 主要出現在需要整合比較或較長 context 的問題。

### 2. 回答品質分析

表現較好的題型是明確、單一欄位或單一產品規格查詢，例如：

- BZH GPU memory 正確回答 `24GB GDDR7`。
- BYH GPU 型號與顯示記憶體正確回答 `RTX 5080 Laptop GPU` 與 `16GB GDDR7`。
- BXH GPU 最大顯示功耗正確回答 `140W`。
- 記憶體容量、類型、速度與插槽可完整回答 `Up to 64GB`、`DDR5`、`5600MHz`、`2x SO-DIMM`。
- BZH、BYH、BXH 的 GPU 型號、VRAM 與功耗差異可以完整列出。
- out-of-scope 問題能拒答，沒有嘗試回答非產品規格問題。

主要不足集中在需要彙整多個欄位或多個 category 的問題：

1. **CPU 題回答不夠完整**

   CPU 問題有抓到 `Intel Core Ultra 9 Processor 275HX`，但沒有補上 expected answer 中的 cache、clock、cores、threads。

2. **Display 完整規格題表現不佳**

   螢幕解析度、更新率、面板類型題中，retrieval 取回的是 color gamut、brightness、features、response time，沒有取回 resolution、refresh rate、OLED 等核心欄位，導致模型回答「目前資料沒有標示」。

3. **Ports 題明顯失敗**

   外接螢幕與高速傳輸問題取回的是 `count` 類型的低資訊 chunk，例如 `right_side.3.count`，模型因此只回答數量，沒有回答 HDMI、Thunderbolt、DisplayPort、Power Delivery 等重點。

4. **多 category 問題容易只回答其中一類**

   無線網路、藍牙、LAN、Webcam 同時詢問時，retrieval 主要取回 Webcam chunks，導致回答只包含 Webcam，Wi-Fi、Bluetooth、LAN 被判斷為未標示。

5. **Negative 題安全但資訊不足**

   5G 行動網路題回答「目前資料沒有標示」是安全的，但沒有補充已知的 Wi-Fi 7、1G LAN、Bluetooth v5.4 作為對照。

整體來看，模型在「retrieval context 已包含正確事實」時能穩定產生正確答案；錯誤多半來自 retrieval 沒有取回足夠完整的 context，或 chunk 粒度過細造成語意不完整。

### 3. RAG Pipeline 分析

目前 pipeline 的優點：

- Chunking 已改為多層結構，能同時產生 fact、shared、specific、comparison chunk。
- Retrieval 對明確型號與單一規格欄位有效，例如 GPU memory、GPU power、Memory、OS。
- Comparison chunk 對多型號 GPU 差異題有幫助。
- Prompt 能限制模型只回答規格問題，out-of-scope 題沒有明顯幻覺。

目前 pipeline 的限制：

- `top_k=4` 對多意圖問題偏少，當問題同時包含多個 category 時容易只取回其中一類。
- Ports 的 fact chunk 過細，取回 `count` 時缺少 port 名稱、版本與支援功能。
- Display 問題需要 category summary 或 field-group chunk，否則單靠 top-k fact chunks 容易漏掉 resolution、panel、refresh rate。
- Connectivity + Webcam 題需要 category-aware retrieval，確保每個被問到的 category 至少有相關 chunk 進入 context。

後續應改善方向：

1. 為 Display、Ports、Connectivity 建立更完整的 category summary chunk。
2. Ports chunk 應保留完整 port 名稱、count、介面版本與支援功能，而不是只切出單一 count。
3. 對多 category 問題增加 category-aware retrieval 或提高 `TOP_K`。
4. 補齊 `data/test_data.json` 的 `required_facts`、`forbidden_facts` 與 `expected_chunks`，讓 evaluation 能計算更客觀的通過率。

---

## 4GB VRAM 使用量驗證

以下分析依據`vram_samples.csv`。

VRAM 取樣摘要：

| 階段 | 峰值 VRAM |
|---|---|
| baseline | 1571 MB |
| idle | 1571 MB |
| load_index | 1571 MB |
| load_embedding | 2041 MB |
| load_llm | 3825 MB |
| retrieve | 3825 MB |
| generate | 3895 MB |
| finished | 3895 MB |

整體峰值：

```text
3895 MB / 15360 MB
```
若以 4GB VRAM 作為限制，這次測試的 3895 MB，低於 4096 MB，因此通過 4GB VRAM 使用量驗證。
