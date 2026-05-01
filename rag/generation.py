import time
from llama_cpp import Llama


#def load_llm(
#    model_path="models/qwen2.5-1.5b.q4_k_m.gguf",
#    n_ctx=2048,
#    n_gpu_layers=20,
#    n_threads=4,
#):
#    return Llama(
#        model_path=model_path,
#        n_ctx=n_ctx,
#        n_gpu_layers=n_gpu_layers,
#        n_threads=n_threads,
#        #verbose=False,
#        verbose=True,
#    )
def load_llm(
    model_path=None,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_threads=4,
    verbose=True,
):
    model_path = model_path or os.getenv("MODEL_PATH")

    if not model_path:
        raise ValueError("MODEL_PATH 未設定，請設定 GGUF 模型路徑")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=verbose,
    )


def get_prompt_content(chunk):
    return chunk.get("content") or chunk.get("text", "")


def build_prompt(query, retrieved_results):
    context = "\n\n".join(
        f"[資料 {i + 1}]\n{get_prompt_content(r['chunk'])}"
        for i, r in enumerate(retrieved_results)
    )

    return f"""你是 GIGABYTE AORUS MASTER 16 AM6H 筆電規格助理。

請只根據下方資料回答，不要補充資料外的內容。
回答規則：
1. 用繁體中文，全英問題則用英文回答，直接回答問題，保持簡短。
2. 不要輸出 JSON key、欄位路徑或英文字段名，例如 dimensions.width。
3. 把規格整理成自然語句，例如「尺寸為 357 mm x 254 mm x 23~29.9 mm，重量約 2.5 kg。」
5. 如果無相關資料，回答「目前無此資料。」並結束回應

資料：
{context}

問題：
{query}

回答：
"""


def generate_stream(llm, prompt, max_tokens=128):
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
            "\n問題：",
            "\n資料：",
            "\n回答：",
            "\n請只根據",
            "\n你是",
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
