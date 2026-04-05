"""
rag_engine.py
=============
Full RAG pipeline:

  1. RETRIEVAL  — Embed query with the same model used in embed.py,
                  search FAISS index, return top-K chunks.
  2. RERANKING  — Simple MMR (Maximal Marginal Relevance) to reduce
                  redundant chunks and improve diversity.
  3. GENERATION — Send retrieved context + query to Groq (free LLM API).

Free stack:
  Embeddings : sentence-transformers/all-MiniLM-L6-v2  (local, no cost)
  Vector DB  : FAISS IndexFlatIP                       (local, no cost)
  LLM        : Groq — llama-3.3-70b-versatile          (free tier)
                Sign up: https://console.groq.com
"""

import os
import json
import numpy as np
import faiss
from typing import Optional
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─── Config ───────────────────────────────────────────────────────────────────
INDEX_FILE   = "data/faiss.index"
META_FILE    = "data/metadata.json"

EMBED_MODEL  = "all-MiniLM-L6-v2"     # must match embed.py
GROQ_MODEL   = "llama-3.3-70b-versatile"

TOP_K        = 10    # Candidates retrieved from FAISS
MMR_K        = 5     # Final chunks kept after MMR reranking
MAX_TOKENS   = 1024
TEMPERATURE  = 0.2

SYSTEM_PROMPT = """You are GitBot 🦊, an expert AI assistant for GitLab employees and candidates.
You answer questions ONLY based on GitLab's official Handbook and Direction pages.

Rules:
- Be accurate, concise, and friendly.
- Cite source URLs when referencing specific content.
- Use markdown: bullet points, **bold**, headers where helpful.
- If the provided context doesn't answer the question, say so clearly — never hallucinate.
- Keep answers focused and practical for someone working at or joining GitLab.
"""


# ─── Retriever ────────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    Semantic retriever using sentence-transformers embeddings + FAISS.
    Supports MMR reranking to reduce redundant results.
    """

    def __init__(
        self,
        index_path: str = INDEX_FILE,
        meta_path:  str = META_FILE,
        model_name: str = EMBED_MODEL,
    ):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'.\n"
                "Run:  python scraper.py  then  python embed.py"
            )
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata not found at '{meta_path}'.\n"
                "Run:  python scraper.py  then  python embed.py"
            )

        print(f"Loading FAISS index from {index_path} …")
        self.index = faiss.read_index(index_path)

        print(f"Loading metadata from {meta_path} …")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        assert self.index.ntotal == len(self.metadata), (
            f"Index has {self.index.ntotal} vectors but metadata has "
            f"{len(self.metadata)} entries — please re-run embed.py"
        )

        print(f"Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)

        print(
            f"Retriever ready — {self.index.ntotal:,} vectors, "
            f"dim={self.index.d}"
        )

    # ── Embed a query ─────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """Returns shape (1, dim) float32 array, L2-normalised."""
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return vec

    # ── Basic top-K search ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Returns top_k chunks most similar to query.
        Each result: {id, url, title, text, score}
        """
        q_vec = self.embed_query(query)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS returns -1 for padding
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── MMR reranking ─────────────────────────────────────────────────────────

    def mmr_rerank(
        self,
        query: str,
        candidates: list[dict],
        final_k: int = MMR_K,
        lambda_param: float = 0.6,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance reranking.

        Balances:
          - Relevance to query       (weight: lambda_param)
          - Diversity from selected  (weight: 1 - lambda_param)

        Higher lambda → more relevance focused.
        Lower  lambda → more diversity focused.
        """
        if not candidates:
            return []
        if len(candidates) <= final_k:
            return candidates

        q_vec = self.embed_query(query)[0]      # (dim,)

        # Embed all candidate texts
        texts = [c["text"] for c in candidates]
        cand_vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")                     # (N, dim)

        # Relevance scores (query ↔ candidate)
        relevance = cand_vecs @ q_vec           # (N,)  cosine via dot (normalised)

        selected_indices: list[int] = []
        remaining        = list(range(len(candidates)))

        while len(selected_indices) < final_k and remaining:
            if not selected_indices:
                # Pick the most relevant first
                best = max(remaining, key=lambda i: relevance[i])
            else:
                # MMR score = λ·relevance − (1−λ)·max_similarity_to_selected
                sel_vecs = cand_vecs[selected_indices]      # (S, dim)
                mmr_scores = []
                for i in remaining:
                    sim_to_selected = float(np.max(cand_vecs[i] @ sel_vecs.T))
                    score = (
                        lambda_param * relevance[i]
                        - (1 - lambda_param) * sim_to_selected
                    )
                    mmr_scores.append((score, i))
                best = max(mmr_scores, key=lambda x: x[0])[1]

            selected_indices.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected_indices]

    # ── Combined retrieve + rerank ────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K, final_k: int = MMR_K) -> list[dict]:
        candidates = self.search(query, top_k=top_k)
        reranked   = self.mmr_rerank(query, candidates, final_k=final_k)
        return reranked


# ─── Generator ───────────────────────────────────────────────────────────────

class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = GROQ_MODEL):
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError(
                "GROQ_API_KEY missing.\n"
                "1. Get a FREE key at https://console.groq.com\n"
                "2. No credit card required.\n"
                "3. Export it:  export GROQ_API_KEY='gsk_...'\n"
                "   Or add to .streamlit/secrets.toml for deployment."
            )
        self.client = Groq(api_key=key)
        self.model  = model
        print(f"Groq generator ready — model: {self.model}")

    def generate(
        self,
        query:         str,
        context_chunks: list[dict],
        chat_history:  Optional[list[dict]] = None,
    ) -> str:
        # Build context block with clear source attribution
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Source {i}] {chunk['title']}\n"
                f"URL: {chunk['url']}\n\n"
                f"{chunk['text']}"
            )
        context_str = "\n\n" + ("─" * 60) + "\n\n".join(context_parts)

        user_msg = (
            f"Use the following context from GitLab's Handbook & Direction pages "
            f"to answer the question.\n\n"
            f"{context_str}\n\n"
            f"{'─' * 60}\n\n"
            f"Question: {query}"
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            messages.extend(chat_history[-8:])   # last 4 turns
        messages.append({"role": "user", "content": user_msg})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content


# ─── Full RAG Pipeline ────────────────────────────────────────────────────────

class GitLabRAG:
    """
    End-to-end pipeline:
      query → FAISS retrieval → MMR rerank → Groq generation → answer
    """

    def __init__(self, api_key: Optional[str] = None):
        self.retriever = FAISSRetriever()
        self.generator = GroqGenerator(api_key=api_key)

    def ask(
        self,
        question:     str,
        chat_history: Optional[list[dict]] = None,
        top_k:        int = TOP_K,
        final_k:      int = MMR_K,
    ) -> dict:
        """
        Returns:
          answer  : str   — LLM response
          sources : list  — unique source URLs cited
          chunks  : list  — retrieved chunk dicts (with scores)
        """
        # 1. Retrieve + rerank
        chunks = self.retriever.retrieve(question, top_k=top_k, final_k=final_k)

        # 2. Generate
        answer = self.generator.generate(question, chunks, chat_history)

        # 3. Collect sources (deduplicated, ordered by first appearance)
        seen    = set()
        sources = []
        for c in chunks:
            if c["url"] not in seen:
                sources.append(c["url"])
                seen.add(c["url"])

        return {
            "answer":  answer,
            "sources": sources,
            "chunks":  chunks,
        }

    @property
    def index_size(self) -> int:
        return self.retriever.index.ntotal


# ─── CLI Quick Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag     = GitLabRAG()
    history = []
    print("\nGitLab RAG chatbot — type 'exit' to quit\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue

        result = rag.ask(q, chat_history=history)
        print(f"\nGitBot: {result['answer']}")
        print("\nSources:")
        for s in result["sources"]:
            print(f"  • {s}")
        print()

        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant",  "content": result["answer"]})
