"""
Hybrid index construction & persistence.

Builds two complementary indexes over the chunk collection:

  * **Lexical (BM25).**  Exact term matching, which is what you want for the
    dense security jargon in this corpus -- "JuicyPotato", "ADCS", "SAM hive",
    "Net-NTLMv1", CVE ids, tool names.  Embeddings tend to blur these; BM25
    nails them.  Primary implementation is `rank_bm25.BM25Okapi`; a compact,
    dependency-free BM25Okapi is included as an offline fallback.

  * **Semantic (embeddings + ANN).**  sentence-transformers + FAISS for
    paraphrase / concept matching ("dump the local password database" ->
    "SAM hive"), with numpy/TF-IDF fallbacks (see backends.py).

Everything is persisted under `cfg.index_dir` so building (slow) and querying
(fast) are separate steps:

    index_store/
      chunks.jsonl        # the chunk records
      bm25.pkl            # tokenised corpus + BM25 params
      vectors.(faiss|npy) # the ANN index
      embedder.pkl        # fitted fallback embedder (only for TF-IDF backend)
      meta.json           # config snapshot + which backends were used
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .backends import build_embedder, build_vector_index, FaissIndex, NumpyFlatIndex
from .chunking import Chunk, chunk_sections
from .config import Config
from .ingestion import ingest_corpus

logger = logging.getLogger("htb_rag.indexing")


# --------------------------------------------------------------------------- #
# Tokenisation for lexical search
# --------------------------------------------------------------------------- #
# Keep alphanumerics together so "juicypotato", "ntlmv1", "adcs", "cve" survive
# as single tokens.  Lowercase for case-insensitive matching.  We deliberately
# do NOT stem: security terms are brittle under stemming.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lexical tokeniser tuned to preserve security jargon."""
    return _TOKEN_RE.findall(text.lower())


# English + question-word stopwords.  We strip these from *queries* only:
# a natural-language question ("how does JuicyPotato work and which machines
# use it?") is mostly filler that dilutes the few high-value jargon tokens in
# BM25.  The corpus side keeps everything -- IDF already down-weights common
# words there, and we don't want to lose legitimate matches.
_STOPWORDS = frozenset("""
a an and are as at be by for from has have how i in is it its of on or that the
to was were what when where which who why with you your can do does did done
me my we our us this these those there here then than so if but not no yes
provide give show tell list explain describe about into using use used via
machine machines box boxes htb write writeup writeups technique techniques
work works working attack attacks exploit exploits method methods way ways
""".split())


def tokenize_query(text: str) -> List[str]:
    """Query-side tokeniser: like `tokenize` but drops stopwords / filler.

    Falls back to the unfiltered tokens if filtering would empty the query
    (e.g. a query that is itself a stopword-like term)."""
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    return toks or _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# BM25 (primary rank_bm25, else pure-python fallback)
# --------------------------------------------------------------------------- #
class PurePythonBM25:
    """
    Minimal BM25Okapi (Robertson/Sparck-Jones) used when rank_bm25 is missing.

    Matches rank_bm25.BM25Okapi's defaults (k1=1.5, b=0.75) so results are
    comparable regardless of which backend is active.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0
        self.doc_freqs: List[dict] = []
        df: dict = {}
        for doc in corpus_tokens:
            freqs: dict = {}
            for t in doc:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in freqs:
                df[t] = df.get(t, 0) + 1
        # BM25Okapi idf with the standard +1 flooring used by rank_bm25.
        self.idf: dict = {}
        for t, n in df.items():
            self.idf[t] = math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5))

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype="float32")
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


def build_bm25(corpus_tokens: List[List[str]], prefer_offline: bool = False):
    if not prefer_offline:
        try:
            from rank_bm25 import BM25Okapi

            logger.info("Using lexical backend: rank_bm25.BM25Okapi")
            return BM25Okapi(corpus_tokens), "rank_bm25.BM25Okapi"
        except Exception as exc:
            logger.warning("rank_bm25 unavailable (%s). Using pure-python BM25.", exc)
    logger.info("Using lexical backend: PurePythonBM25 (offline fallback)")
    return PurePythonBM25(corpus_tokens), "PurePythonBM25"


# --------------------------------------------------------------------------- #
# The persisted index object
# --------------------------------------------------------------------------- #
@dataclass
class HybridIndex:
    """In-memory handle to a built index; created by `build_index` / `load_index`."""
    cfg: Config
    chunks: List[Chunk]
    bm25: object
    bm25_backend: str
    corpus_tokens: List[List[str]]
    embedder: object
    vector_index: object
    meta: dict

    # ---- persistence -----------------------------------------------------
    def save(self) -> None:
        d = self.cfg.index_dir
        d.mkdir(parents=True, exist_ok=True)

        # 1) chunks
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

        # 2) BM25 (store tokenised corpus + backend name; rebuild on load so we
        #    do not depend on pickling a third-party object across versions)
        with open(d / "bm25.pkl", "wb") as fh:
            pickle.dump(
                {"corpus_tokens": self.corpus_tokens, "backend": self.bm25_backend},
                fh,
            )

        # 3) vector index
        if isinstance(self.vector_index, FaissIndex):
            self.vector_index.save(str(d / "vectors.faiss"))
            vec_backend = "faiss"
        else:  # NumpyFlatIndex
            self.vector_index.save(str(d / "vectors.npy"))
            vec_backend = "numpy"

        # 4) fitted fallback embedder (TF-IDF) needs to be persisted; the
        #    sentence-transformer one is reloaded from its model name.
        from .backends import TfidfSvdEmbedder

        if isinstance(self.embedder, TfidfSvdEmbedder):
            with open(d / "embedder.pkl", "wb") as fh:
                pickle.dump(self.embedder, fh)
            emb_backend = "tfidf-svd"
        else:
            emb_backend = "sentence-transformers"

        # 5) meta
        meta = dict(self.meta)
        meta.update({
            "config": self.cfg.to_dict(),
            "bm25_backend": self.bm25_backend,
            "vector_backend": vec_backend,
            "embedder_backend": emb_backend,
            "embedding_model": self.cfg.embedding_model,
            "embedding_dim": int(getattr(self.embedder, "dim", 0)),
            "num_chunks": len(self.chunks),
        })
        with open(d / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Index saved to %s (%d chunks)", d, len(self.chunks))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_index(cfg: Config, prefer_offline: bool = False,
                show_progress: bool = True) -> HybridIndex:
    """Full build: ingest -> chunk -> BM25 + embeddings -> persist-ready object."""
    logger.info("Ingesting corpus from %s ...", cfg.corpus_dir)
    sections = ingest_corpus(cfg)
    logger.info("Parsed %d sections", len(sections))

    chunks = chunk_sections(sections, cfg)
    logger.info("Produced %d chunks", len(chunks))

    # ---- Lexical index ---------------------------------------------------
    corpus_tokens = [tokenize(c.text) for c in chunks]
    bm25, bm25_backend = build_bm25(corpus_tokens, prefer_offline=prefer_offline)

    # ---- Semantic index --------------------------------------------------
    embedder, needs_fit = build_embedder(
        cfg.embedding_model, cfg.normalize_embeddings, prefer_offline=prefer_offline)
    texts = [c.text for c in chunks]
    if needs_fit:
        embedder.fit(texts)          # corpus-fitted fallback embedder
    embeddings = embedder.encode(
        texts, batch_size=cfg.embedding_batch_size, show_progress=show_progress)

    vector_index = build_vector_index(embedder.dim, prefer_offline=prefer_offline)
    vector_index.add(embeddings)

    meta = {"num_sections": len(sections)}
    return HybridIndex(
        cfg=cfg, chunks=chunks, bm25=bm25, bm25_backend=bm25_backend,
        corpus_tokens=corpus_tokens, embedder=embedder,
        vector_index=vector_index, meta=meta)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_index(cfg: Config) -> HybridIndex:
    """Reconstruct a HybridIndex from `cfg.index_dir`."""
    d = cfg.index_dir
    if not (d / "meta.json").exists():
        raise FileNotFoundError(
            f"No index found in {d}. Run `python -m htb_rag.cli index` first.")

    meta = json.loads((d / "meta.json").read_text())

    # chunks
    chunks: List[Chunk] = []
    with open(d / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            chunks.append(Chunk.from_dict(json.loads(line)))

    # BM25 (rebuild from stored tokens using the same backend selection logic)
    bm25_blob = pickle.loads((d / "bm25.pkl").read_bytes())
    corpus_tokens = bm25_blob["corpus_tokens"]
    prefer_offline = bm25_blob["backend"] == "PurePythonBM25"
    bm25, bm25_backend = build_bm25(corpus_tokens, prefer_offline=prefer_offline)

    # embedder
    if meta.get("embedder_backend") == "tfidf-svd":
        embedder = pickle.loads((d / "embedder.pkl").read_bytes())
    else:
        from .backends import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(
            meta["embedding_model"], normalize=cfg.normalize_embeddings)

    # vector index
    if meta.get("vector_backend") == "faiss":
        vector_index = FaissIndex.load(str(d / "vectors.faiss"))
    else:
        vector_index = NumpyFlatIndex.load(str(d / "vectors.npy"))

    return HybridIndex(
        cfg=cfg, chunks=chunks, bm25=bm25, bm25_backend=bm25_backend,
        corpus_tokens=corpus_tokens, embedder=embedder,
        vector_index=vector_index, meta=meta)
