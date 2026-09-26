# HTB Cheatsheet Assistant — Hybrid RAG

A retrieval-augmented generation system that answers natural-language questions about
offensive-security techniques by retrieving relevant passages from a corpus of
[Hack The Box write-ups](https://github.com/0xh7ml/htb-wiki) and having an LLM synthesize a
**grounded, cited** answer — never a generic answer from the model's own training data.

Reference query it is built to handle:

> *"Provide me the Windows privilege escalation cheatsheet."*

...returns techniques grouped by family (SAM/SYSTEM hive dumps, potato-family exploits,
ADCS abuse, ACL/BloodHound abuse, …), each with a short explanation and a machine citation
like `(seen on: Absolute, APT)`.

---

## 1. How it works

```
raw *.md  ─►  ingestion  ─►  chunking  ─►  ┌──────────────┐   ┌───────────────┐
 (513 HTB     (sections,     (section-     │ BM25 (lexical)│   │ vectors (FAISS│
  write-ups)   metadata)      aware,        │  rank_bm25    │   │ + MiniLM)     │
                              overlap)      └──────┬───────┘   └───────┬───────┘
                                                   │  candidates        │
                                                   └──────►  RRF fusion ◄┘
                                                                │
                                              MMR + per-machine cap (diversity)
                                                                │
                                                     top-k chunks ─► LLM synthesis
                                                                      (context-only,
                                                                       forced citations)
```

| Stage | Module | What it does |
|---|---|---|
| Ingestion | `htb_rag/ingestion.py` | Parse markdown → `{machine_name, section_title, content, source_path, stage}`. Code-fence-aware; robust machine-name resolution; fallback for missing/odd headers. |
| Chunking | `htb_rag/chunking.py` | Section-aware chunks, sub-split long sections to ~300–500 tokens with 50-token overlap. Machine + section metadata on **every** chunk. |
| Indexing | `htb_rag/indexing.py` + `backends.py` | BM25 (`rank_bm25`) + dense embeddings (`sentence-transformers`) in FAISS. Persisted to disk. |
| Retrieval | `htb_rag/retrieval.py` | Hybrid BM25 + vector, **Reciprocal Rank Fusion**, **MMR** diversity + per-machine cap. |
| Expansion | `htb_rag/query_expansion.py` | Broad "cheatsheet" queries → focused per-technique sub-queries. |
| Synthesis | `htb_rag/synthesis.py` | Claude / OpenAI call. Context-only system prompt; mandatory `(seen on: X)` citations. |
| Evaluation | `htb_rag/evaluation.py` | Recall / precision / hit / MRR vs. a hand-derived key. |

### Graceful degradation (important)

Every heavy dependency has a **pure fallback** so the pipeline runs end-to-end with only
`numpy` + `scikit-learn`:

| Production | Fallback (offline) |
|---|---|
| `rank_bm25.BM25Okapi` | pure-python BM25Okapi (identical defaults) |
| `sentence-transformers` MiniLM | TF-IDF + Truncated-SVD dense embedder (sklearn) |
| FAISS `IndexFlatIP` | numpy brute-force inner-product |

The active backend is logged at build time and stored in `index_store/meta.json`.

---

## 2. File structure

```
htb_rag_project/
├── README.md
├── requirements.txt
├── htb_rag/                 # the package
│   ├── config.py            # all tunables + section taxonomy
│   ├── ingestion.py         # markdown → structured sections
│   ├── chunking.py          # section-aware chunking + overlap
│   ├── backends.py          # embedder + vector-index backends (prod + fallback)
│   ├── indexing.py          # BM25 + vectors, build/save/load
│   ├── retrieval.py         # hybrid RRF + MMR
│   ├── query_expansion.py   # broad-query decomposition
│   ├── synthesis.py         # LLM synthesis, grounding + citations
│   ├── pipeline.py          # expand → retrieve → synthesize glue
│   ├── evaluation.py        # metrics harness
│   └── cli.py               # command-line entrypoint
├── data/
│   ├── testset.json         # 16 questions + hand-derived gold machines
│   └── eval_report.json     # detailed per-question scores
├── scripts/
│   └── build_answer_key.py  # derives gold sets by grepping raw files
└── docs/
    ├── evaluation.md        # 1-page evaluation writeup
    └── design_note.md       # 1/2-page design note
```

The raw corpus is **not** vendored. Clone it next to the project (see below).

---

## 3. Setup

### 3.1 Dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Minimum to run offline (fallback backends): `numpy`, `scikit-learn`.
Full production stack: add `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `tiktoken`.
For synthesis: `anthropic` and/or `openai`.

### 3.2 Get the corpus

```bash
git clone https://github.com/0xh7ml/htb-wiki.git
# raw write-ups are in htb-wiki/raw/  (513 *.md files)
```

### 3.3 Configuration

Everything is in `htb_rag/config.py` and can be overridden by environment variable:

| Env var | Meaning | Default |
|---|---|---|
| `HTB_CORPUS_DIR` | path to raw `*.md` | `htb-wiki/raw` |
| `HTB_INDEX_DIR` | where the index is stored | `index_store` |
| `HTB_EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` |
| `HTB_LLM_PROVIDER` | `gemini` \| `anthropic` \| `openai` \| `none` | `anthropic` |
| `ANTHROPIC_API_KEY` | Claude key (synthesis) | — |
| `OPENAI_API_KEY` | OpenAI key (synthesis) | — |
| `GEMINI_API_KEY` | Google Gemini key — **free tier** (synthesis) | — |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `OPENAI_BASE_URL` | override for any OpenAI-compatible endpoint (Groq, Ollama) | — |

Key retrieval knobs (edit `config.py`): `final_top_k`, `rrf_k`, `mmr_lambda`,
`max_chunks_per_machine`, `bm25_top_k`, `vector_top_k`.

### Free LLM synthesis with Google Gemini

Synthesis works with Anthropic, OpenAI, **or Google Gemini's free tier**.
Gemini is reached through its OpenAI-compatible endpoint, so it uses the `openai` SDK:

```bash
pip install openai
export GEMINI_API_KEY="AIza...your_key..."     # from https://aistudio.google.com (Get API key)
python -m htb_rag.cli query "Provide me the Windows privilege escalation cheatsheet." --provider gemini
```

That's it — no other env vars needed (base URL and model have sensible defaults). The
`ANSWER` block becomes synthesized, grouped prose with `(seen on: Machine)` citations and
the metadata line reads `provider=gemini used_llm=True`. If the key/model is unavailable the
system logs the reason (`-v`) and falls back to the extractive digest instead of crashing.

---

## 4. Usage — exact commands

All commands are run from the project root.

### 4.1 Build the index (once)

```bash
# Production backends (needs rank_bm25 + sentence-transformers + faiss):
python -m htb_rag.cli index --corpus htb-wiki/raw --index-dir index_store -v

# Force offline fallback backends (only numpy + scikit-learn required):
python -m htb_rag.cli index --corpus htb-wiki/raw --offline -v
```

### 4.2 Ask a question

```bash
# The reference cheatsheet query (auto-expanded into sub-queries):
python -m htb_rag.cli query "Provide me the Windows privilege escalation cheatsheet."

# A specific question:
python -m htb_rag.cli query "How does JuicyPotato work and which machines use it?" --k 6

# See retrieved context only (no LLM):
python -m htb_rag.cli query "Which machines dump the SAM and SYSTEM hive?" --retrieval-only --show-snippets

# Choose a provider explicitly (Gemini free tier shown):
GEMINI_API_KEY=AIza... python -m htb_rag.cli query "ADCS ESC1 cheatsheet" --provider gemini
```

Without an API key the answer is an **extractive, citation-preserving digest** of the
retrieved context, so the pipeline is fully demonstrable offline.

### 4.3 Evaluate

```bash
# Rebuild the hand-derived answer key from the raw files (optional):
python scripts/build_answer_key.py --corpus htb-wiki/raw --questions data/testset.json --out data/testset.json

# Score retrieval:
python -m htb_rag.cli eval --testset data/testset.json --k 8 --out data/eval_report.json
```

### 4.4 Inspect the built index

```bash
python -m htb_rag.cli info
```

---

## 5. Example output (reference query, offline backends)

```
$ python -m htb_rag.cli query "Provide me the Windows privilege escalation cheatsheet." --retrieval-only
[*] Expanded into 9 sub-queries.
=== RETRIEVED CONTEXT ================================================
[1] Bastion   | Dump Hashes From Registry   (stage=foothold)   ← SAM/SYSTEM hive
[2] Authority | Create Certificate          (stage=privesc)    ← ADCS
[3] Conceal   | Watson                       (stage=privesc)   ← kernel exploit suggester
[4] Omni      | Extract Hashes               (stage=privesc)   ← SAM hive
[5] Cicada    | Via reg / secretsdump        (stage=privesc)   ← SAM hive
...
```

With an API key the synthesis step turns this into grouped prose, e.g.:

```
**SAM / SYSTEM hive dump** — copy the SAM and SYSTEM registry hives and run
secretsdump locally to recover NT hashes. (seen on: Bastion, Omni, Cicada)

**ADCS certificate abuse (ESC1)** — enrol a certificate for a privileged
principal against a misconfigured template, then pass-the-cert. (seen on: Authority)
...
```

---

## 6. Deliverables map (assessment)

| Requirement | Where |
|---|---|
| Working code: ingestion / indexing / retrieval / synthesis | `htb_rag/` |
| README with run instructions | this file |
| ~15 test questions + hand-derived answer key | `data/testset.json` (16 Qs) + `scripts/build_answer_key.py` |
| Evaluation writeup (recall/precision + synthesis quality) | `docs/evaluation.md` |
| Design note (chunking, lexical vs embedding, next step) | `docs/design_note.md` |

> **Note on the evaluation environment:** the committed scores were produced with the
> **offline fallback embedder** (no network to download `sentence-transformers` weights).
> The architecture under test is the full hybrid; only the embedding model is a weaker
> stand-in, so the reported precision (0.86) / hit-rate (1.00) / MRR (0.97) are a lower
> bound. See `docs/evaluation.md` for the full analysis and ablation.
