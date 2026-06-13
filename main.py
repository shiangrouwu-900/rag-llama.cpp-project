# main.py
import os
from pathlib import Path

from rag.embedding import load_embedding_model
from rag.retrieval import load_index, retrieve
from rag.generation import load_llm, build_prompt, generate_stream
from rag.build_index import main as build_index_main


STORAGE_DIR = "storage"
DEFAULT_MODEL_PATH = "models/Qwen3-1.7B-Q4_K_M.gguf"
TOP_K = 4


def index_files_exist(storage_dir=STORAGE_DIR):
    storage_path = Path(storage_dir)
    return (
        (storage_path / "chunks.json").exists()
        and (storage_path / "embeddings.npy").exists()
    )


def should_rebuild_index():
    """
    預設不重建 index。
    如果需要強制重建，可在執行前設定：
    REBUILD_INDEX=1 uv run python -m main
    """
    return os.getenv("REBUILD_INDEX", "0") == "1"


def prepare_index(storage_dir=STORAGE_DIR):
    if should_rebuild_index() or not index_files_exist(storage_dir):
        print("找不到既有 index，或已設定 REBUILD_INDEX=1，開始建立 index...")
        build_index_main()

    print("載入 index...")
    embeddings, chunks = load_index(storage_dir)
    print(f"已載入 chunks: {len(chunks)}")

    return embeddings, chunks


def get_model_path():
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"找不到模型檔案：{model_path}\n"
            "請設定 MODEL_PATH，例如：\n"
            'export MODEL_PATH="/content/drive/MyDrive/rag_models/Qwen3-1.7B-Q4_K_M.gguf"'
        )

    return model_path


def main():
    embeddings, chunks = prepare_index(STORAGE_DIR)

    print("載入 embedding model...")
    embedding_model = load_embedding_model()

    print("載入 LLM...")
    llm = load_llm(get_model_path())

    print("AI 助手已啟動。")
    print("你可以詢問 GIGABYTE AORUS MASTER 16 AM6H 產品規格。")
    print("輸入 exit / quit / q 離開。")

    while True:
        query = input("\n請輸入問題：").strip()

        if query.lower() in {"exit", "quit", "q"}:
            print("結束程式。")
            break

        if not query:
            print("請輸入有效問題。")
            continue

        results = retrieve(
            query=query,
            model=embedding_model,
            embeddings=embeddings,
            chunks=chunks,
            top_k=TOP_K,
        )

        prompt = build_prompt(query, results)

        print("\n回答：")
        generate_stream(llm, prompt)
        print()


if __name__ == "__main__":
    main()
