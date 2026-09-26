"""
Pluggable backends for embeddings and vector search.

The assessment specifies the *production* stack:

    * embeddings : sentence-transformers  (all-MiniLM-L6-v2)
    * vector ANN : FAISS                  (IndexFlatIP, cosine via inner product)
    * lexical    : rank_bm25              (BM25Okapi)   -- see indexing.py

Those are the primary implementations selected whenever the libraries are
importable.  Because CI / air-gapped boxes frequently cannot download the
model weights or the wheels, each backend has a dependency-free **fallback**
built on numpy / scikit-learn so the whole pipeline stays runnable and
testable offline.  The active backend is recorded in the index metadata and
printed at build time, so there is never any ambiguity about what produced a
set of results.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("htb_rag.backends")


# ========================================================================== #
#  Embedding backends
# ========================================================================== #
class BaseEmbedder:
    name: str = "base"
    dim: int = 0

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Primary, production embedder (spec-mandated)."""

    def __init__(self, model_name: str, normalize: bool = True):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=64, show_progress=False):
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vecs.astype("float32")


class TfidfSvdEmbedder(BaseEmbedder):
    """
    Offline fallback embedder.

    TF-IDF over word + char n-grams, reduced to a dense vector with Truncated
    SVD (LSA).  This is a genuine dense semantic embedding (captures term
    co-occurrence / synonymy at the corpus level) and needs no network or GPU,
    only scikit-learn.  It is deterministic given a fixed seed, which makes the
    offline evaluation reproducible.
    """

    def __init__(self, dim: int = 256, normalize: bool = True, seed: int = 42):
        self.name = f"tfidf-svd:{dim}"
        self.requested_dim = dim
        self.normalize = normalize
        self.seed = seed
        self._vectorizer = None
        self._svd = None
        self.dim = dim

    def fit(self, corpus: List[str]) -> "TfidfSvdEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        )
        tfidf = self._vectorizer.fit_transform(corpus)
        n_comp = min(self.requested_dim, tfidf.shape[1] - 1, max(2, tfidf.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        self._svd.fit(tfidf)
        self.dim = n_comp
        return self

    def encode(self, texts, batch_size=64, show_progress=False):
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfSvdEmbedder.fit() must be called before encode().")
        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf).astype("float32")
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


def build_embedder(model_name: str, normalize: bool, prefer_offline: bool = False):
    """
    Select an embedder.  Returns (embedder, needs_fit).

    `needs_fit` is True for the corpus-fitted fallback so the caller knows to
    call `.fit(corpus)` before encoding.
    """
    if not prefer_offline:
        try:
            emb = SentenceTransformerEmbedder(model_name, normalize=normalize)
            logger.info("Using embedder: %s (dim=%d)", emb.name, emb.dim)
            return emb, False
        except Exception as exc:  # ImportError or model-download failure
            logger.warning(
                "sentence-transformers unavailable (%s). "
                "Falling back to offline TF-IDF+SVD embedder.", exc)

    emb = TfidfSvdEmbedder(normalize=normalize)
    logger.info("Using embedder: %s (offline fallback)", emb.name)
    return emb, True


# ========================================================================== #
#  Vector index backends
# ========================================================================== #
class BaseVectorIndex:
    name = "base"

    def add(self, vectors: np.ndarray) -> None: ...
    def search(self, queries: np.ndarray, top_k: int): ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BaseVectorIndex": ...


class FaissIndex(BaseVectorIndex):
    """Primary vector index: FAISS IndexFlatIP (exact cosine on unit vectors)."""

    def __init__(self, dim: int):
        import faiss  # lazy

        self._faiss = faiss
        self.name = "faiss:IndexFlatIP"
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        self.index.add(np.ascontiguousarray(vectors.astype("float32")))

    def search(self, queries, top_k):
        scores, idxs = self.index.search(
            np.ascontiguousarray(queries.astype("float32")), top_k)
        return scores, idxs

    def save(self, path):
        self._faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path, dim=None):
        import faiss

        obj = cls.__new__(cls)
        obj._faiss = faiss
        obj.name = "faiss:IndexFlatIP"
        obj.index = faiss.read_index(path)
        obj.dim = obj.index.d
        return obj


class NumpyFlatIndex(BaseVectorIndex):
    """
    Offline fallback: exact brute-force inner-product search in numpy.

    Identical results to FAISS IndexFlatIP for unit-normalised vectors; only
    slower at very large scale (fine for a 10k-chunk corpus like this one).
    """

    def __init__(self, dim: int):
        self.name = "numpy:flat-ip"
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None

    def add(self, vectors):
        v = vectors.astype("float32")
        self._vectors = v if self._vectors is None else np.vstack([self._vectors, v])

    def search(self, queries, top_k):
        if self._vectors is None:
            raise RuntimeError("Index is empty.")
        sims = queries.astype("float32") @ self._vectors.T  # (nq, N)
        top_k = min(top_k, sims.shape[1])
        idxs = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        # Sort the top_k slice by score descending.
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, idxs], axis=1)
        idxs = idxs[rows, order]
        scores = sims[rows, idxs]
        return scores, idxs

    def save(self, path):
        np.save(path, self._vectors)

    @classmethod
    def load(cls, path, dim=None):
        vecs = np.load(path)
        obj = cls(vecs.shape[1])
        obj._vectors = vecs
        return obj


def build_vector_index(dim: int, prefer_offline: bool = False) -> BaseVectorIndex:
    if not prefer_offline:
        try:
            idx = FaissIndex(dim)
            logger.info("Using vector index: %s", idx.name)
            return idx
        except Exception as exc:
            logger.warning("FAISS unavailable (%s). Using numpy flat index.", exc)
    idx = NumpyFlatIndex(dim)
    logger.info("Using vector index: %s (offline fallback)", idx.name)
    return idx
