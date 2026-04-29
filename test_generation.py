# test_generation.py
import os
from rag.generation import load_llm, build_prompt, generate_stream

cloud_model_path = os.getenv("MODEL_PATH")

def main():
    llm = load_llm(
        model_path=cloud_model_path,
        n_ctx=2048,
        n_gpu_layers=20,
        n_threads=4,
    )

    query = "AORUS MASTER 16 AM6H 的 CPU 是什麼？"

    # 先用假資料測 generation，不依賴 retrieval
    retrieved_results = [
        {
            "score": 0.95,
            "chunk": {
                "chunk_id": "AM6H_CPU",
                "text": """Models: AORUS MASTER 16 AM6H
Category: Processor
CPU: Intel Core Ultra 9 Processor 275HX
Features: AI PC, high performance processor""",
                "metadata": {
                    "models": ["AORUS MASTER 16 AM6H"],
                    "category": "Processor",
                },
            },
        }
    ]

    prompt = build_prompt(query, retrieved_results)

    print("=" * 80)
    print("Prompt:")
    print(prompt)
    print("=" * 80)
    print("Answer:")

    output_text, metrics = generate_stream(
        llm=llm,
        prompt=prompt,
        max_tokens=128,
    )

    print("\n" + "=" * 80)
    print("Metrics:")
    print(metrics)

    if output_text.strip():
        print("\nGeneration test: PASS")
    else:
        print("\nGeneration test: FAIL - no output")


if __name__ == "__main__":
    main()
