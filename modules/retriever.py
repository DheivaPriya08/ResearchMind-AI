"""
retriever.py
------------
FAISS-backed semantic search index over paper chunks.

Supports single-paper search (Week 1 deliverable) and multi-paper /
cross-paper search (Week 3 deliverable) using the same underlying index --
multi-paper search is just single-paper search without a paper_id filter,
plus an optional post-filter.
"""

from dataclasses import dataclass
import numpy as np
import faiss

from modules.embeddings import embed_texts, embed_query
from modules.pdf_loader import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


class PaperIndex:
    """An in-memory FAISS index over chunks from one or more papers.

    Design note: this is intentionally a single flat index shared across all
    loaded papers (each chunk carries its own paper_id), rather than one
    index per paper. That makes "search across all my papers" the default
    case and "search within one paper" a filtered query -- matching how the
    project actually gets used (load several papers, then ask questions that
    may or may not care which paper the answer comes from).
    """

    def __init__(self):
        self.index = None  # faiss.IndexFlatIP, built lazily once we know dim
        self.chunks: list = []  # parallel array: chunks[i] <-> index row i
        self.dim = None

    def add_chunks(self, chunks: list):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)
        if self.index is None:
            self.dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def search(self, query: str, top_k: int = 5, paper_id: str = None) -> list:
        """Semantic search. If paper_id is given, restricts results to that
        paper (by over-fetching from FAISS then filtering, since FAISS's flat
        index doesn't support metadata filtering natively)."""
        if self.is_empty():
            return []

        qvec = embed_query(query)
        fetch_k = top_k * 6 if paper_id else top_k  # over-fetch to survive filtering
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(qvec, fetch_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if paper_id and chunk.paper_id != paper_id:
                continue
            results.append(SearchResult(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def papers_loaded(self) -> list:
        """Distinct (paper_id, title) pairs currently in the index."""
        seen = {}
        for c in self.chunks:
            seen[c.paper_id] = c.paper_title
        return [(pid, title) for pid, title in seen.items()]

    def chunks_for_paper(self, paper_id: str) -> list:
        return [c for c in self.chunks if c.paper_id == paper_id]
