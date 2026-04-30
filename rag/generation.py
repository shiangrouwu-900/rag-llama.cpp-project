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

    return f"""請遵守以下規則回答：

1. 只回答 GIGABYTE AORUS MASTER 16 AM6H、BZH、BYH、BXH 的產品規格問題。如果問題與產品規格無關，請只回答「我只能回答規格問題」。
2. 請嚴格根據下方「產品資料」回答，不要使用外部知識，不要自行推測，不要補充資料中沒有的內容。
3. 如果「產品資料」中沒有足夠資訊回答，請回答：「根據目前資料無法確認」。
4. 如果問題詢問的是 BZH、BYH、BXH 共同規格，請根據 shared 規格回答。
5. 如果問題明確指定 BZH、BYH 或 BXH，請優先回答該型號的 specific 規格。
6. 如果問題詢問 BZH、BYH、BXH 的差異、比較、哪裡不同，請優先根據 comparison 規格回答。
7. 詢問 GPU 但未指定型號時，若資料中有 BZH、BYH、BXH 的 GPU 比較，請回答三個型號差異，不要只回答單一型號。
8. alias、相關詞、常見問句只用來理解使用者問題，不可以把 alias 當成產品規格答案。
9. 請使用繁體中文回答。如果使用者問題是全英文，才使用英文回答。

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
        "請問還有其他問題",
        "如果資料中沒有答案",
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
