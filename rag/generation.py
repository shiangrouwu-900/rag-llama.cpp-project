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

    return f"""你是 GIGABYTE AORUS MASTER 16 AM6H 產品規格 AI 助手。

請嚴格根據下方「產品資料」回答使用者問題。
如果資料中沒有答案，請回答「根據目前資料無法確認」，不要自行編造。
如果遇到跟產品規格無關的問題，請回答「我只能回答規格問題」，不要回應無關問題。
請使用繁體中文回答；若使用者用英文提問，也可以用英文回答。

產品資料：
{context}

使用者問題：
{query}

回答：
"""


def generate_stream(llm, prompt, max_tokens=256):
    start_time = time.perf_counter()
    first_token_time = None
    output_text = ""
    token_count = 0

    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        stop=["使用者問題：", "\n\n使用者："],
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
