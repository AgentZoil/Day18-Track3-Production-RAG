"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    _shared_model = None
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if CrossEncoderReranker._shared_model is None:
            from FlagEmbedding import FlagReranker

            CrossEncoderReranker._shared_model = FlagReranker(
                self.model_name,
                use_fp16=True
            )

            # warmup
            CrossEncoderReranker._shared_model.compute_score([("test", "test")])

        return CrossEncoderReranker._shared_model
    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        # TODO: Implement reranking
        # 1. model = self._load_model()
        # 2. pairs = [(query, doc["text"]) for doc in documents]
        # 3. scores = model.compute_score(pairs)  # FlagReranker
        #    OR scores = model.predict(pairs)      # CrossEncoder
        # 4. Combine: [(score, doc) for score, doc in zip(scores, documents)]
        # 5. Sort by score descending
        # 6. Return top_k RerankResult(text=..., original_score=doc["score"],
        #                              rerank_score=score, metadata=doc["metadata"], rank=i)
        if not documents:
            return []

        model = self._load_model()

        # 1. Create query-doc pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # 2. Compute rerank scores
        scores = model.compute_score(pairs)

        # 3. Combine
        scored_docs = []
        for score, doc in zip(scores, documents):
            scored_docs.append({
                "text": doc["text"],
                "original_score": doc.get("score", 0.0),
                "rerank_score": float(score),
                "metadata": doc.get("metadata", {})
            })

        # 4. Sort descending by rerank_score
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 5. Build result
        results = []
        for i, doc in enumerate(scored_docs[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc["original_score"],
                rerank_score=doc["rerank_score"],
                metadata=doc["metadata"],
                rank=i + 1
            ))

        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        from flashrank import RerankRequest

        passages = [{"text": d["text"]} for d in documents]

        results = self._model.rerank(
            RerankRequest(query=query, passages=passages)
        )

        reranked = []
        for i, r in enumerate(results[:top_k]):
            doc = documents[r["corpus_id"]]
            reranked.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=r["score"],
                metadata=doc.get("metadata", {}),
                rank=i + 1
            ))



def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    # TODO: Implement benchmark
    # 1. times = []
    # 2. for _ in range(n_runs):
    #      start = time.perf_counter()
    #      reranker.rerank(query, documents)
    #      times.append((time.perf_counter() - start) * 1000)  # ms
    # 3. return {"avg_ms": mean(times), "min_ms": min(times), "max_ms": max(times)}
    from statistics import mean
    times = []

    # Warm-up (important!)
    reranker.rerank(query, documents)

    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "avg_ms": round(mean(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
