"""
embed.py
========
Generates sentence embeddings for all chunks and saves a FAISS index.

Model  : all-MiniLM-L6-v2  (free, runs locally, 80MB, very fast)
Index  : FAISS IndexFlatIP  (inner-product / cosine similarity)

Saves:
  data/faiss.index     — FAISS binary index
  data/metadata.json   — chunk metadata aligned to FAISS index rows

Usage:
  python embed.py

This only needs to be run once (or when you re-scrape).
Takes ~1–5 min for 500 pages on CPU.
"""

import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ─── Config ───────────────────────────────────────────────────────────────────
CHUNKS_FILE  = "data/chunks.json"
INDEX_FILE   = "data/faiss.index"
META_FILE    = "data/metadata.json"

# Free, lightweight, no GPU needed — 384-dim embeddings
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 64           # Chunks per embedding batch
SHOW_EVERY   = 200          # Progress print interval


# ─── Load Chunks ──────────────────────────────────────────────────────────────

def load_chunks(path: str = CHUNKS_FILE) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.\nRun scraper first:  python scraper.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ─── Embed ───────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list, model_name: str = EMBED_MODEL) -> np.ndarray:
    """
    Returns float32 numpy array of shape (N, embedding_dim).
    Vectors are L2-normalised so dot-product == cosine similarity.
    """
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    N     = len(texts)
    print(f"Embedding {N} chunks in batches of {BATCH_SIZE} …\n")

    all_embeddings = []
    t0 = time.time()

    for i in range(0, N, BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise for cosine via dot
        )
        all_embeddings.append(embeddings)

        if (i // BATCH_SIZE) % max(1, SHOW_EVERY // BATCH_SIZE) == 0:
            elapsed = time.time() - t0
            done    = min(i + BATCH_SIZE, N)
            print(f"  [{done:>5}/{N}]  {elapsed:.1f}s elapsed")

    matrix = np.vstack(all_embeddings).astype("float32")
    print(f"\nEmbedding matrix: {matrix.shape}  ({matrix.nbytes / 1e6:.1f} MB)")
    return matrix


# ─── Build FAISS Index ────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP = exact inner-product search.
    Since vectors are L2-normalised, IP == cosine similarity.
    For large corpora (>100K chunks) swap to IndexIVFFlat for speed.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built  |  dim={dim}  |  vectors={index.ntotal:,}")
    return index


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_index(index: faiss.Index, path: str = INDEX_FILE):
    faiss.write_index(index, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"FAISS index saved  -> {path}  ({size_mb:.1f} MB)")


def save_metadata(chunks: list, path: str = META_FILE):
    """Save only the metadata fields (no text duplicated in index)."""
    meta = [{"id": c["id"], "url": c["url"], "title": c["title"], "text": c["text"]}
            for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(path) / 1024
    print(f"Metadata saved     -> {path}  ({size_kb:.0f} KB, {len(meta)} records)")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    chunks     = load_chunks()
    embeddings = embed_chunks(chunks)
    index      = build_faiss_index(embeddings)

    save_index(index)
    save_metadata(chunks)

    print("\nAll done!")
    print(f"  FAISS index : {INDEX_FILE}")
    print(f"  Metadata    : {META_FILE}")
    print("\nNext step: python app.py   OR   streamlit run app.py")
