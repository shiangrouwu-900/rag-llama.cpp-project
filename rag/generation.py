import time
from llama_cpp import Llama



def load_llm(
    model_path="models/qwen2.5-1.5b.q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=20,
    n_threads=4,
):
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )


def build_prompt(query, retrieved_results):
    context = "\n\n".join(
        f"[資料 {i+1}]\n{r['chunk']['text']}"
        for i, r in enumerate(retrieved_results)
    )

    return f"""你是產品規格問答系統。

限制：
只能根據產品資料回答。
資料不足時，只輸出：根據目前資料無法確認
非產品規格問題時，只輸出：我只能回答規格問題
禁止反問。禁止自我檢查。禁止重複規則。禁止輸出答案以外的文字。

產品資料：
{context}

使用者問題：
{query}

請只輸出最終答案。不要解釋規則、不要反問、不要補充資料中沒有的內容、不要重複句子。

回答：
"""


def generate_stream(llm, prompt, max_tokens=64):
    start_time = time.perf_counter()
    first_token_time = None
    output_text = ""
    token_count = 0

    stream = llm(
    prompt,
    max_tokens=max_tokens,
    temperature=0.0,
    top_p=1.0,
    stop=[
        "\n使用者問題：",
        "\n產品資料：",
        "\n回答：",
        "\n請確認",
        "請確認回答是否符合規則",
        "是的。",
        "如果問題與產品規格無關",
        "如果「產品資料」",
        "請問還有其他問題",
    ],
    stream=True,
)

    for chunk in stream:
        token = chunk["choices"][0]["text"]

        if token:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            print(token, end="", flush=True)
            output_text += token
            token_count += 1

    end_time = time.perf_counter()
    generation_time = end_time - (first_token_time or start_time)

    metrics = {
        "ttft": first_token_time - start_time if first_token_time else None,
        "tps": token_count / generation_time if generation_time > 0 else 0,
        "output_tokens": token_count,
    }

    return output_text, metrics
