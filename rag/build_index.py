# build_index.py
import json
from pathlib import Path

import numpy as np

from chunking import build_chunks, load_product_data
from embedding import load_embedding_model


PRODUCT_DATA_PATH = Path("data/product_info.json")
OUTPUT_DIR = Path("storage")
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"


def get_chunk_text(chunk, *, purpose="embedding"):
    """Read text from both the old and new chunk formats."""
    preferred_keys = (
        ("search_text", "content", "text")
        if purpose == "embedding"
        else ("content", "text", "search_text")
    )

    for key in preferred_keys:
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value

    chunk_id = chunk.get("id", "<unknown>")
    raise ValueError(f"Chunk {chunk_id} has no usable text/content/search_text field.")


def normalize_chunks(chunks):
    """Keep chunks compatible with retrieval, generation, and evaluation code."""
    normalized = []

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"Chunk #{i} must be a dict, got {type(chunk).__name__}.")

        item = dict(chunk)
        prompt_text = get_chunk_text(item, purpose="prompt")
        embedding_text = get_chunk_text(item, purpose="embedding")

        item.setdefault("id", f"chunk:{i}")
        item.setdefault("text", prompt_text)
        item.setdefault("content", prompt_text)
        item.setdefault("search_text", embedding_text)

        normalized.append(item)

    return normalized


def build_embeddings(chunks, model):
    texts = [get_chunk_text(chunk, purpose="embedding") for chunk in chunks]
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def save_index(chunks, embeddings, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / CHUNKS_FILE
    embeddings_path = output_dir / EMBEDDINGS_FILE

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(embeddings_path, embeddings)

    return chunks_path, embeddings_path


def main():
    print("Loading product data...")
    product_data = load_product_data(PRODUCT_DATA_PATH)

    print("Building chunks...")
    chunks = normalize_chunks(build_chunks(product_data))
    print(f"Chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Building embeddings from search_text/content/text...")
    embeddings = build_embeddings(chunks, embedding_model)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Saving index...")
    chunks_path, embeddings_path = save_index(chunks, embeddings, OUTPUT_DIR)
    print(f"Saved chunks: {chunks_path}")
    print(f"Saved embeddings: {embeddings_path}")
    print("Done.")


if __name__ == "__main__":
    main()
