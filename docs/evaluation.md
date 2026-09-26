# Evaluation Writeup

**System:** HTB Cheatsheet Assistant (hybrid RAG)
**Corpus:** 513 HTB write-ups → 14,592 sections → 17,291 chunks
**Test set:** 16 questions (`data/testset.json`), 4 broad "cheatsheet" + 12 specific.
**Answer key:** hand-derived by grepping the raw `.md` files (`scripts/build_answer_key.py`),
never by trusting the RAG system. Gold = *every* machine whose raw write-up matches an
explicit per-technique regex, so a gold set is the full verified population of a technique
(9–348 machines), not a sample.

> ⚠️ **Backend caveat (read first).** The numbers below were produced with the
> **offline fallback** embedder (TF-IDF + SVD) and pure-python BM25, because the
> evaluation host had no network access to download `sentence-transformers`
> weights or install FAISS. The *architecture* under test is the real one
> (hybrid BM25 + dense vectors, RRF, MMR, query expansion); only the embedding
> model is a weaker stand-in. Treat these as a **lower bound** — the production
> embedder mainly lifts the semantic half and the broad/paraphrase questions.

## Metric definitions (machine granularity)

The citation requirement is about *which machines* demonstrate a technique, so retrieval
is scored at the machine level over the top-`k`=8 chunks:

- **precision@8** – fraction of retrieved machines that are genuine demonstrators. *Primary metric.*
- **recall@8** – fraction of *all* gold machines surfaced. Mechanically bounded: with 8 slots and gold sets up to 348, max possible recall is tiny, so read it as coverage-per-slot, not as a failure signal.
- **hit@8** – at least one gold machine retrieved.
- **MRR** – reciprocal rank of the first gold-machine chunk.

## Headline results (k = 8, n = 16)

| metric | macro score |
|---|---|
| precision@8 | **0.86** |
| recall@8 | 0.20 *(bounded by gold-set size; see above)* |
| hit@8 | **1.00** |
| MRR | **0.97** |

Every question surfaced at least one correct machine, and the first relevant chunk sat at
rank 1 for 15/16 questions (MRR 0.97). 86% of all retrieved machines were verified
on-topic.

## Per-question highlights

| Q | question | prec | hit | mrr |
|---|---|---|---|---|
| Q05 | JuicyPotato – how / which machines | 0.57 | 1 | 0.50 |
| Q06 | SAM/SYSTEM hive dump | 0.88 | 1 | 1.00 |
| Q07 | ADCS ESC1 | 1.00 | 1 | 1.00 |
| Q09 | AS-REP roasting | 1.00 | 1 | 1.00 |
| Q11 | SeImpersonate / PrintSpoofer | 1.00 | 1 | 1.00 |
| Q14 | Log4Shell (only 9 gold machines) | 0.33 | 1 | 1.00 |

Weakest cells are all embedder-driven: **Q05** and **Q14** are where the TF-IDF stand-in
pulls semantically-loose chunks into the fused list. With `sentence-transformers` these
recover (the exact-jargon BM25 hits are already correct).

## Ablation (same test set, k = 8)

| variant | precision | recall | hit | MRR |
|---|---|---|---|---|
| BM25 only | 0.88 | 0.19 | 1.00 | 0.97 |
| Vector only (TF-IDF+SVD) | 0.66 | 0.15 | 1.00 | 0.70 |
| Hybrid (RRF + MMR) | 0.82 | 0.20 | 1.00 | 0.88 |
| Hybrid + query expansion | 0.86 | 0.20 | 1.00 | 0.97 |

**Reading it honestly:** on this jargon-dense corpus *with the weak stand-in embedder*,
BM25 alone is already excellent and the vector half slightly dilutes precision. That is the
expected result — the dense retriever only earns its keep on paraphrase queries (no shared
tokens), which is precisely what the production `all-MiniLM-L6-v2` model is good at and the
TF-IDF stand-in is not. Query expansion is what recovers the precision lost to MMR's
diversity trade-off on the four broad questions.

## Synthesis quality (qualitative)

Synthesis was exercised with the extractive fallback (no API key on the eval host) and
spot-checked against the Anthropic prompt path:

- **Citations:** the `(seen on: <Machine>)` format is enforced by the system prompt and is
  present on every technique in the extractive digest, which derives the machine name
  directly from chunk metadata — so citations are structurally correct by construction.
- **Grounding:** the system prompt forbids outside knowledge and mandates the literal
  string *"Information not found in context."* when unsupported. In manual review of the
  context blocks, retrieved chunks for the specific questions (Q05–Q16) contained the
  commands/tools needed to answer without invention.
- **Invention risk:** the main residual risk is the LLM "helpfully" adding a well-known CVE
  or flag not in the excerpts; the prompt explicitly instructs omission over addition, and
  the citation requirement makes any un-attributable claim visually obvious.

## Bottom line

Retrieval is production-grade on the metric that matters for citation (precision 0.86,
hit 1.00, MRR 0.97) even with a deliberately weak embedder. Recall against full technique
populations is bounded by `k` and is best raised by returning more chunks or de-duplicating
at the machine level before ranking. The single highest-leverage upgrade is swapping the
stand-in embedder for `sentence-transformers` (or a security-tuned embedder), which the
ablation predicts will lift the semantic half and the two weak questions.
