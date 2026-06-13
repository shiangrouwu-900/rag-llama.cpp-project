import time
import subprocess
import sys
import tempfile
from pathlib import Path

from llama_cpp import Llama


class LlamaCliRunner:
    def __init__(
        self,
        model_path,
        n_ctx=3072,
        n_gpu_layers=0,
        n_threads=4,
        cli_path="tools/llama.cpp/llama-completion.exe",
    ):
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.cli_path = str(cli_path)

        if not Path(self.cli_path).exists():
            raise FileNotFoundError(f"找不到 llama-cli：{self.cli_path}")

    def __call__(
        self,
        prompt,
        max_tokens=192,
        temperature=0.0,
        top_p=1.0,
        stop=None,
        stream=True,
    ):
        stop = stop or []
        prompt_file = None

        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
                f.write(prompt)
                prompt_file = f.name

            command = [
                self.cli_path,
                "-m",
                self.model_path,
                "-f",
                prompt_file,
                "-n",
                str(max_tokens),
                "-c",
                str(self.n_ctx),
                "-t",
                str(self.n_threads),
                "-ngl",
                str(self.n_gpu_layers),
                "--temp",
                str(temperature),
                "--top-p",
                str(top_p),
                "--no-display-prompt",
                "-no-cnv",
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
                "--no-perf",
                "--no-warmup",
                "--color",
                "off",
            ]

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )

            if process.returncode != 0:
                raise RuntimeError(process.stderr.strip() or f"llama-cli exited with code {process.returncode}")

            output = clean_cli_output(process.stdout, stop)
            if output:
                yield {"choices": [{"text": output}]}
        finally:
            if prompt_file:
                Path(prompt_file).unlink(missing_ok=True)


def clean_cli_output(output, stop):
    text = output.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    for marker in ["[end of text]", "<|im_end|>", "<|endoftext|>"]:
        text = text.replace(marker, "")

    for marker in stop or []:
        if marker and marker in text:
            text = text.split(marker)[0]

    return text.strip()


def load_llm(
    model_path="models/Qwen3-1.7B-Q4_K_M.gguf",
    n_ctx=3072,
    n_gpu_layers=0,
    n_threads=4,
):
    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
    except Exception as exc:
        print(f"llama-cpp-python 載入失敗，改用 llama-cli CPU fallback：{exc}")
        return LlamaCliRunner(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
        )


def get_prompt_content(chunk):
    return chunk.get("content") or chunk.get("text", "")


def build_prompt(query, retrieved_results):
    if not retrieved_results:
        context = "沒有取回到可回答此問題的產品規格資料。若使用者問題不是產品規格問題，請回答「我只能回答產品規格問題」。"
    else:
        context = "\n\n".join(
            f"[資料 {i + 1}]\n{get_prompt_content(result['chunk'])}"
            for i, result in enumerate(retrieved_results)
        )

    return f"""/no_think
你是產品規格問答助手。你只能根據使用者提供的「資料」回答。

回答規則：
0. 不要輸出推理過程，不要輸出 <think>，只輸出最終答案。
1. 只回答問題問到的型號、規格類別或欄位，不要補充無關規格。
2. 單一欄位問題請直接回答欄位值，保持簡短。
3. 比較問題請用型號條列，並說明相同或不同。
4. 如果資料沒有明確標示，回答「目前資料沒有標示」。
5. 如果問題不是產品規格問題，回答「我只能回答產品規格問題」。
6. 不要輸出 JSON key、內部欄位路徑或程式欄位名稱，除非使用者明確要求。
7. 使用繁體中文回答；若問題完全是英文，使用英文回答。

資料：
{context}

問題：{query}
答案：
"""


def generate_stream(llm, prompt, max_tokens=192):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
            "<|im_end|>",
            "<|im_start|>",
            "\n問題：",
            "\n資料：",
            "\n回答：",
        ],
        stream=True,
    )

    for chunk in stream:
        token = chunk["choices"][0]["text"]

        if token:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            output_text += token
            token_count += 1

    end_time = time.perf_counter()
    generation_time = end_time - (first_token_time or start_time)
    output_text = clean_cli_output(output_text, [])

    if output_text:
        print(output_text, end="", flush=True)

    metrics = {
        "ttft": first_token_time - start_time if first_token_time else None,
        "tps": token_count / generation_time if generation_time > 0 else 0,
        "output_tokens": token_count,
    }

    return output_text, metrics
