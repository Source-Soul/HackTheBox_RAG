"""
Hybrid retrieval: BM25 + vector, fused, then diversified.

Pipeline for one query:

    1. Lexical stage   -> top-N BM25 candidates
    2. Semantic stage  -> top-N vector candidates
    3. Fusion          -> Reciprocal Rank Fusion (default) or weighted score
    4. Diversity       -> MMR rerank + a hard per-machine cap

Why fuse at all?  The two signals fail in opposite ways.  BM25 is unbeatable on
exact jargon ("JuicyPotato", "ADCS") but blind to paraphrase; embeddings catch
paraphrase but wash out rare literal tokens.  RRF combines their *rankings*
(not their incomparable raw scores), which is robust and parameter-light.

Why diversify?  A broad "cheatsheet" query otherwise returns eight chunks from
the two or three write-ups that happen to mention the topic most often.  The
user wants *coverage* -- many techniques across many machines -- so MMR trades
a little relevance for novelty, and a per-machine cap guarantees breadth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chunking import Chunk
from .config import Config
from .indexing import HybridIndex, tokenize_query


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float                       # final (post-fusion) score
    bm25_rank: Optional[int] = None    # 1-based rank in lexical list (None = absent)
    vector_rank: Optional[int] = None  # 1-based rank in semantic list
    bm25_score: float = 0.0
    vector_score: float = 0.0


# --------------------------------------------------------------------------- #
# Stage 1+2: candidate generation
# --------------------------------------------------------------------------- #
def _bm25_candidates(index: HybridIndex, query: str, top_k: int) -> List[Tuple[int, float]]:
    scores = np.asarray(index.bm25.get_scores(tokenize_query(query)), dtype="float32")
    if scores.size == 0:
        return []
    k = min(top_k, scores.size)
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top]


def _vector_candidates(index: HybridIndex, query: str, top_k: int) -> List[Tuple[int, float]]:
    qvec = index.embedder.encode([query])
    scores, idxs = index.vector_index.search(qvec, top_k)
    out: List[Tuple[int, float]] = []
    for i, s in zip(idxs[0], scores[0]):
        if i >= 0:
            out.append((int(i), float(s)))
    return out


# --------------------------------------------------------------------------- #
# Stage 3: fusion
# --------------------------------------------------------------------------- #
def _reciprocal_rank_fusion(
    bm25: List[Tuple[int, float]],
    vector: List[Tuple[int, float]],
    k: int,
) -> Dict[int, float]:
    """RRF: score(d) = sum_r 1 / (k + rank_r(d)), ranks are 1-based."""
    fused: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(vector, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def _minmax(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _weighted_fusion(
    bm25: List[Tuple[int, float]],
    vector: List[Tuple[int, float]],
    alpha: float,
) -> Dict[int, float]:
    """Weighted sum of min-max-normalised scores: alpha*vec + (1-alpha)*bm25."""
    bm = _minmax({i: s for i, s in bm25})
    ve = _minmax({i: s for i, s in vector})
    fused: Dict[int, float] = {}
    for i in set(bm) | set(ve):
        fused[i] = alpha * ve.get(i, 0.0) + (1 - alpha) * bm.get(i, 0.0)
    return fused


# --------------------------------------------------------------------------- #
# Stage 4: MMR diversity + per-machine cap
# --------------------------------------------------------------------------- #
def _mmr_rerank(
    index: HybridIndex,
    candidate_idxs: List[int],
    relevance: Dict[int, float],
    query: str,
    cfg: Config,
) -> List[int]:
    """
    Maximal Marginal Relevance over the fused candidate pool.

        MMR = lambda * rel(d) - (1 - lambda) * max_sim(d, already_selected)

    Similarity uses the chunk embeddings already in the index (recomputed here
    for the small candidate pool).  A per-machine cap is applied on top so no
    single box can occupy more than `cfg.max_chunks_per_machine` slots.
    """
    if not candidate_idxs:
        return []

    # Embed the candidate chunks + the query for pairwise similarity.
    cand_texts = [index.chunks[i].text for i in candidate_idxs]
    cand_vecs = index.embedder.encode(cand_texts)
    qvec = index.embedder.encode([query])[0]

    # Normalise rel scores to [0,1] so lambda trades off cleanly against sim.
    rel_norm = _minmax({i: relevance[i] for i in candidate_idxs})

    # Precompute query relevance as a fallback ordering signal.
    sim_to_query = cand_vecs @ qvec
    pos = {idx: p for p, idx in enumerate(candidate_idxs)}

    selected: List[int] = []
    machine_counts: Dict[str, int] = {}
    remaining = set(candidate_idxs)
    lam = cfg.mmr_lambda

    while remaining and len(selected) < cfg.final_top_k:
        best_idx = None
        best_score = -1e9
        for idx in remaining:
            machine = index.chunks[idx].machine_slug
            # Hard diversity guarantee: skip if this machine is already full,
            # unless we have run out of other machines to choose from.
            if machine_counts.get(machine, 0) >= cfg.max_chunks_per_machine:
                other_available = any(
                    index.chunks[j].machine_slug != machine
                    or machine_counts.get(index.chunks[j].machine_slug, 0)
                    < cfg.max_chunks_per_machine
                    for j in remaining
                )
                if other_available:
                    continue

            rel = rel_norm.get(idx, 0.0)
            if selected:
                sims = cand_vecs[pos[idx]] @ cand_vecs[[pos[s] for s in selected]].T
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0
            mmr = lam * rel - (1 - lam) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
        m = index.chunks[best_idx].machine_slug
        machine_counts[m] = machine_counts.get(m, 0) + 1

    return selected


def _apply_machine_cap_only(index: HybridIndex, ordered_idxs: List[int],
                            cfg: Config) -> List[int]:
    """Diversity path when MMR is disabled: keep fused order, cap per machine."""
    selected, counts = [], {}
    for idx in ordered_idxs:
        m = index.chunks[idx].machine_slug
        if counts.get(m, 0) >= cfg.max_chunks_per_machine:
            continue
        selected.append(idx)
        counts[m] = counts.get(m, 0) + 1
        if len(selected) >= cfg.final_top_k:
            break
    return selected


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
class Retriever:
    """Stateful retriever bound to a built HybridIndex."""

    def __init__(self, index: HybridIndex):
        self.index = index
        self.cfg = index.cfg

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        cfg = self.cfg
        final_k = top_k or cfg.final_top_k

        bm25 = _bm25_candidates(self.index, query, cfg.bm25_top_k)
        vector = _vector_candidates(self.index, query, cfg.vector_top_k)

        # Fusion.
        if cfg.fusion_method == "weighted":
            fused = _weighted_fusion(bm25, vector, cfg.weighted_alpha)
        else:
            fused = _reciprocal_rank_fusion(bm25, vector, cfg.rrf_k)

        if not fused:
            return []

        # Rank lookup tables for explainability.
        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25, start=1)}
        vector_rank = {idx: r for r, (idx, _) in enumerate(vector, start=1)}
        bm25_score = {idx: s for idx, s in bm25}
        vector_score = {idx: s for idx, s in vector}

        # Take the fused top pool for reranking.
        pool = sorted(fused, key=lambda i: fused[i], reverse=True)[: cfg.mmr_pool]

        if cfg.use_mmr:
            chosen = _mmr_rerank(self.index, pool, fused, query, cfg)
        else:
            chosen = _apply_machine_cap_only(self.index, pool, cfg)

        chosen = chosen[:final_k]

        results: List[RetrievedChunk] = []
        for idx in chosen:
            results.append(RetrievedChunk(
                chunk=self.index.chunks[idx],
                score=float(fused[idx]),
                bm25_rank=bm25_rank.get(idx),
                vector_rank=vector_rank.get(idx),
                bm25_score=float(bm25_score.get(idx, 0.0)),
                vector_score=float(vector_score.get(idx, 0.0)),
            ))
        return results

    def retrieve_multi(self, queries: List[str],
                       top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Multi-query retrieval: run each (sub-)query, RRF-merge their fused
        rankings across queries, then diversify once.  Used for broad
        "cheatsheet" questions after query expansion.
        """
        cfg = self.cfg
        final_k = top_k or cfg.final_top_k

        merged: Dict[int, float] = {}
        bm25_rank: Dict[int, int] = {}
        vector_rank: Dict[int, int] = {}
        bm25_score: Dict[int, float] = {}
        vector_score: Dict[int, float] = {}

        for q in queries:
            bm25 = _bm25_candidates(self.index, q, cfg.bm25_top_k)
            vector = _vector_candidates(self.index, q, cfg.vector_top_k)
            if cfg.fusion_method == "weighted":
                fused = _weighted_fusion(bm25, vector, cfg.weighted_alpha)
            else:
                fused = _reciprocal_rank_fusion(bm25, vector, cfg.rrf_k)
            # MAX-fusion across sub-queries (not sum): we want the chunks that
            # are the *best hit for some technique family*, not the chunks that
            # appear moderately across all of them (generic enumeration pages).
            for idx, s in fused.items():
                merged[idx] = max(merged.get(idx, 0.0), s)
            # Keep the best (lowest) rank / max score seen across sub-queries.
            for r, (idx, s) in enumerate(bm25, start=1):
                if idx not in bm25_rank or r < bm25_rank[idx]:
                    bm25_rank[idx] = r
                bm25_score[idx] = max(bm25_score.get(idx, 0.0), s)
            for r, (idx, s) in enumerate(vector, start=1):
                if idx not in vector_rank or r < vector_rank[idx]:
                    vector_rank[idx] = r
                vector_score[idx] = max(vector_score.get(idx, 0.0), s)

        if not merged:
            return []

        pool = sorted(merged, key=lambda i: merged[i], reverse=True)[: cfg.mmr_pool]
        # MMR needs a single "query" anchor; use the first (original) query.
        if cfg.use_mmr:
            chosen = _mmr_rerank(self.index, pool, merged, queries[0], cfg)
        else:
            chosen = _apply_machine_cap_only(self.index, pool, cfg)
        chosen = chosen[:final_k]

        results: List[RetrievedChunk] = []
        for idx in chosen:
            results.append(RetrievedChunk(
                chunk=self.index.chunks[idx],
                score=float(merged[idx]),
                bm25_rank=bm25_rank.get(idx),
                vector_rank=vector_rank.get(idx),
                bm25_score=float(bm25_score.get(idx, 0.0)),
                vector_score=float(vector_score.get(idx, 0.0)),
            ))
        return results

    # Convenience: return the raw fused ranking without diversity, for eval.
    def retrieve_fused_only(self, query: str, top_k: int) -> List[int]:
        cfg = self.cfg
        bm25 = _bm25_candidates(self.index, query, cfg.bm25_top_k)
        vector = _vector_candidates(self.index, query, cfg.vector_top_k)
        if cfg.fusion_method == "weighted":
            fused = _weighted_fusion(bm25, vector, cfg.weighted_alpha)
        else:
            fused = _reciprocal_rank_fusion(bm25, vector, cfg.rrf_k)
        return sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
