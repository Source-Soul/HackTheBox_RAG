"""
Central configuration for the HTB Cheatsheet RAG system.

Every tunable knob lives here so the rest of the code stays declarative.
Values can be overridden via environment variables (see `Config.from_env`)
which keeps secrets (API keys) out of the source tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Section taxonomy
# --------------------------------------------------------------------------- #
# HTB write-ups (0xdf style) do NOT use a consistent "PrivEsc" / "Root" header.
# In practice the corpus uses headings like:
#   "Shell as root", "Shell as SYSTEM", "Shell as Administrator",
#   "Beyond Root", "PrivEsc to root", "Priv: www-data -> root", "Root.txt" ...
# We map raw heading text onto a small set of logical *stages* so that
# retrieval and evaluation can reason about "privilege escalation" content
# regardless of the exact wording an author used.
#
# The ordering matters: patterns are tried top-to-bottom, first match wins.
SECTION_STAGE_PATTERNS = [
    # (stage, regex fragment matched case-insensitively against heading text)
    ("privesc", r"\bbeyond\s+root\b"),
    ("privesc", r"\bshell\s+as\s+(root|system|administrator|admin)\b"),
    ("privesc", r"\bprivesc\b|\bpriv\s*esc\b|\bprivilege\s+escalation\b"),
    ("privesc", r"\bpriv\s*:\s*"),           # "Priv: www-data -> root"
    ("privesc", r"\broot\.txt\b|\broot\s+flag\b|\bgetting\s+root\b"),
    ("privesc", r"->\s*(root|system|administrator)\b"),
    ("foothold", r"\bshell\s+as\b|\bfoothold\b|\binitial\s+access\b|\buser\.txt\b"),
    ("foothold", r"\bgetting\s+a?\s*shell\b|\brce\b|\bremote\s+code\b"),
    ("recon", r"\bbox\s+info\b|\brecon\b|\benumeration\b|\bnmap\b|\bscan\b"),
    ("web", r"\bwebsite\b|\bweb\s*-\s*port\b|\bport\s+80\b|\bport\s+443\b"),
]

# Headings we consider pure metadata / boilerplate and drop from retrieval.
# "Box Info" is present in all 513 files and contains only difficulty/OS/dates.
BOILERPLATE_HEADINGS = {"box info"}


# --------------------------------------------------------------------------- #
# Main config object
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # ---- Paths -----------------------------------------------------------
    # Directory containing the raw *.md write-ups.
    corpus_dir: Path = Path("htb-wiki/raw")
    # Directory where the built index (chunks + BM25 + FAISS) is persisted.
    index_dir: Path = Path("index_store")

    # ---- Chunking --------------------------------------------------------
    # Target sub-chunk size and overlap, expressed in *tokens*.
    chunk_target_tokens: int = 400        # centre of the 300-500 band
    chunk_max_tokens: int = 500           # hard ceiling before we force a split
    chunk_min_tokens: int = 60            # merge tiny trailing fragments upward
    chunk_overlap_tokens: int = 50        # sliding-window overlap

    # ---- Embeddings / vector index --------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    normalize_embeddings: bool = True     # cosine similarity via inner product

    # ---- Retrieval -------------------------------------------------------
    bm25_top_k: int = 40                  # candidates from lexical stage
    vector_top_k: int = 40                # candidates from semantic stage
    rrf_k: int = 60                       # Reciprocal Rank Fusion constant
    fusion_method: str = "rrf"            # "rrf" or "weighted"
    weighted_alpha: float = 0.5           # weight on vector score if weighted
    final_top_k: int = 8                  # chunks handed to the LLM

    # ---- Diversity (MMR) -------------------------------------------------
    use_mmr: bool = True
    mmr_lambda: float = 0.7               # 1.0 = pure relevance, 0 = pure diversity
    mmr_pool: int = 30                    # rerank this many fused candidates
    max_chunks_per_machine: int = 2       # hard cap so one box can't dominate

    # ---- Synthesis / LLM -------------------------------------------------
    llm_provider: str = "anthropic"       # "anthropic" | "openai" | "none"
    anthropic_model: str = "claude-sonnet-4-5"
    openai_model: str = "gpt-4o"
    # Any OpenAI-compatible endpoint (Groq, local Ollama, etc.); None = real OpenAI.
    openai_base_url: Optional[str] = None
    # Google Gemini via its OpenAI-compatible endpoint (free tier, no card).
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_max_tokens: int = 1500
    llm_temperature: float = 0.0          # deterministic, grounded synthesis
    # API keys are read from the environment, never hard-coded.
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # ---- Misc ------------------------------------------------------------
    random_seed: int = 42

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config, layering (1) defaults, (2) env vars, (3) kwargs."""
        cfg = cls()

        # Path overrides
        if os.getenv("HTB_CORPUS_DIR"):
            cfg.corpus_dir = Path(os.environ["HTB_CORPUS_DIR"])
        if os.getenv("HTB_INDEX_DIR"):
            cfg.index_dir = Path(os.environ["HTB_INDEX_DIR"])

        # Model overrides
        if os.getenv("HTB_EMBEDDING_MODEL"):
            cfg.embedding_model = os.environ["HTB_EMBEDDING_MODEL"]
        if os.getenv("HTB_LLM_PROVIDER"):
            cfg.llm_provider = os.environ["HTB_LLM_PROVIDER"]

        # Secrets
        cfg.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        cfg.openai_api_key = os.getenv("OPENAI_API_KEY")
        # Gemini key: accept GEMINI_API_KEY or Google's GOOGLE_API_KEY.
        cfg.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # Optional endpoint / model overrides for OpenAI-compatible providers.
        if os.getenv("OPENAI_BASE_URL"):
            cfg.openai_base_url = os.environ["OPENAI_BASE_URL"]
        if os.getenv("OPENAI_MODEL"):
            cfg.openai_model = os.environ["OPENAI_MODEL"]
        if os.getenv("GEMINI_MODEL"):
            cfg.gemini_model = os.environ["GEMINI_MODEL"]

        # Explicit programmatic overrides win last.
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown config field: {key}")
            setattr(cfg, key, value)

        # Coerce path-like strings to Path.
        cfg.corpus_dir = Path(cfg.corpus_dir)
        cfg.index_dir = Path(cfg.index_dir)
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        # Paths and secrets are not JSON-friendly / not safe to dump.
        d["corpus_dir"] = str(self.corpus_dir)
        d["index_dir"] = str(self.index_dir)
        d.pop("anthropic_api_key", None)
        d.pop("openai_api_key", None)
        d.pop("gemini_api_key", None)
        return d
