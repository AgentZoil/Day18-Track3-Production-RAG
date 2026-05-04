"""Module 2: Hybrid Search.

Implements a production-style retrieval layer with:
- Vietnamese-aware BM25 tokenization
- Dense retrieval backed by Qdrant when available
- Deterministic local fallbacks for offline/test environments
- Reciprocal Rank Fusion to merge sparse and dense rankings
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass(slots=True)
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into space-delimited tokens.

    `underthesea` is used when available because it handles Vietnamese word
    boundaries better than a naive whitespace split. A regex fallback keeps the
    module functional in minimal environments.
    """

    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""

    try:
        from underthesea import word_tokenize

        return word_tokenize(cleaned, format="text")
    except Exception:
        tokens = re.findall(r"[\wÀ-ỹ]+", cleaned, flags=re.UNICODE)
        return " ".join(tokens)


def _normalize_text(text: str) -> str:
    """Normalize text before tokenization for BM25 and fallback embeddings."""
    return segment_vietnamese(text).lower().strip()


def _as_numpy(vector: Any) -> np.ndarray:
    """Convert encoder outputs to a 1D numpy array."""
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32, copy=False)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return np.asarray(vector, dtype=np.float32)


class _FallbackBM25:
    """Minimal BM25 implementation for environments without rank_bm25."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        n_docs = len(corpus_tokens)

        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc in self.corpus_tokens:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1

            doc_len = len(doc) or 1
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def _result_identity(result: SearchResult) -> str:
    """Build a stable identity for a retrieved chunk.

    The text alone is not always enough because the same text may appear in
    multiple chunks. We prefer explicit metadata identifiers when available.
    """

    metadata = result.metadata or {}
    identity = {
        "text": result.text,
        "source": metadata.get("source"),
        "parent_id": metadata.get("parent_id"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_type": metadata.get("chunk_type"),
        "id": metadata.get("id"),
    }
    return json.dumps(identity, ensure_ascii=False, sort_keys=True)


def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)
    return float(np.dot(a, b) / denom)


class BM25Search:
    """BM25 index/search over Vietnamese tokenized chunks."""

    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build the BM25 index from text chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = []
        for chunk in self.documents:
            text = chunk.get("text", "")
            tokenized = _normalize_text(text).split()
            self.corpus_tokens.append(tokenized)

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = _FallbackBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search the BM25 index and return the top-k hits."""
        top_k = max(0, int(top_k))
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = _normalize_text(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    text=doc.get("text", ""),
                    score=float(scores[idx]),
                    metadata=dict(doc.get("metadata", {})),
                    method="bm25",
                )
            )
        return results


class _FallbackEncoder:
    """Deterministic text encoder used when sentence-transformers is unavailable."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def encode(self, texts, show_progress_bar: bool = False):  # noqa: ARG002
        if isinstance(texts, str):
            return self._encode_one(texts)
        return np.vstack([self._encode_one(text) for text in texts])

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _normalize_text(text).split()
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def _prepare_payload(chunk: dict) -> dict:
    """Create a Qdrant payload / local metadata mirror for a chunk."""
    payload = dict(chunk.get("metadata", {}))
    payload["text"] = chunk.get("text", "")
    return payload


class DenseSearch:
    """Dense retrieval backed by Qdrant with an in-memory fallback."""

    def __init__(self):
        self.client = None
        self._collections: dict[str, list[dict]] = {}
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            self.client = None
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Prefer a local cache first so offline environments do not
                # spend time retrying external downloads. If the model is not
                # cached, fall back to the deterministic encoder unless the
                # caller explicitly opts into downloading.
                self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                if os.environ.get("ALLOW_HF_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                    except Exception:
                        self._encoder = _FallbackEncoder(dim=EMBEDDING_DIM)
                else:
                    self._encoder = _FallbackEncoder(dim=EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant and keep a local mirror for fallback use."""
        if not chunks:
            self._collections[collection] = []
            return

        texts = [c.get("text", "") for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        vectors = np.asarray(vectors, dtype=np.float32)

        if self.client is not None:
            try:
                from qdrant_client.models import Distance, PointStruct, VectorParams

                self.client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                )
                points = []
                for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
                    payload = _prepare_payload(chunk)
                    points.append(
                        PointStruct(
                            id=i,
                            vector=_as_numpy(vector).tolist(),
                            payload=payload,
                        )
                    )
                if points:
                    self.client.upsert(collection_name=collection, points=points)
                # Keep a local mirror so search still works if the server later
                # becomes unavailable.
                self._collections[collection] = [
                    {
                        "text": chunk.get("text", ""),
                        "metadata": dict(chunk.get("metadata", {})),
                        "vector": _as_numpy(vector),
                    }
                    for chunk, vector in zip(chunks, vectors, strict=False)
                ]
                return
            except Exception:
                # Fall back to in-memory indexing if Qdrant is unavailable.
                self.client = None

        self._collections[collection] = []
        for chunk, vector in zip(chunks, vectors, strict=False):
            self._collections[collection].append(
                {
                    "text": chunk.get("text", ""),
                    "metadata": dict(chunk.get("metadata", {})),
                    "vector": _as_numpy(vector),
                }
            )

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search the dense index and return the top-k hits."""
        top_k = max(0, int(top_k))
        query_vector = _as_numpy(self._get_encoder().encode(query))

        if self.client is not None:
            try:
                hits = self.client.search(
                    collection_name=collection,
                    query_vector=query_vector.tolist(),
                    limit=top_k,
                )
                results: list[SearchResult] = []
                for hit in hits:
                    payload = dict(hit.payload or {})
                    results.append(
                        SearchResult(
                            text=payload.get("text", ""),
                            score=float(hit.score),
                            metadata=payload,
                            method="dense",
                        )
                    )
                return results
            except Exception:
                # If the collection is missing or the server is offline, use
                # the local fallback index if available.
                pass

        docs = self._collections.get(collection, [])
        if not docs:
            return []

        q = query_vector
        scored = []
        for doc in docs:
            vec = doc["vector"]
            score = _cosine_score(q, vec)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            payload = {**doc["metadata"], "text": doc["text"]}
            results.append(
                SearchResult(
                    text=doc["text"],
                    score=float(score),
                    metadata=payload,
                    method="dense",
                )
            )
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    top_k = max(0, int(top_k))
    rrf_scores: dict[str, dict[str, object]] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = _result_identity(result)
            entry = rrf_scores.setdefault(
                key,
                {
                    "score": 0.0,
                    "result": result,
                },
            )
            entry["score"] = float(entry["score"]) + 1.0 / (k + rank + 1)

    merged = []
    for entry in rrf_scores.values():
        base: SearchResult = entry["result"]  # type: ignore[assignment]
        merged.append(
            SearchResult(
                text=base.text,
                score=float(entry["score"]),
                metadata=dict(base.metadata),
                method="hybrid",
            )
        )

    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
