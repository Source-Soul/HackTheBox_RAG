# Design Note

## Chunking strategy

The corpus is 513 0xdf-style write-ups. Two properties drove the design:

1. **Sections are natural technique units.** Headings like `Shell as SYSTEM`,
   `Dump Hashes From Registry`, `ESC13 Background` each wrap one self-contained
   technique. So chunking is **section-aware first**: parse the markdown into
   header-scoped sections and keep a section whole whenever it fits the token
   budget. Only over-long sections are sub-chunked into ~400-token windows
   (300–500 band) with a 50-token overlap, and the sub-split is **fence-aware**
   so a shell transcript is never cut in half.
2. **The markdown is adversarial to naive parsers.** ~2,800 lines beginning
   with `#` live *inside* code fences (root prompts, bash comments). The parser
   tracks ``` / ~~~ state and only splits on real headings — otherwise those
   become thousands of junk sections. Missing/odd headers are handled too:
   prose before the first heading becomes an `Overview` section, and headings
   are classified into logical stages (recon/web/foothold/privesc) with
   inheritance, so `Shell as root → RunAsCs` is tagged `privesc` even though the
   leaf heading is uninformative.

Every chunk carries `machine_name` + `section_title` as metadata **and** as a
prepended text header (`[Machine: X | Section: Y]`). The header makes the
machine name a first-class BM25 token and grounds the LLM's citation.

## Lexical vs. embedding retrieval — why both

This corpus is unusually **jargon-dense**: `JuicyPotato`, `ADCS ESC1`,
`SeImpersonatePrivilege`, `secretsdump -sam`, CVE ids. These are exactly the
tokens dense embeddings blur together, and exactly what BM25 nails. Conversely,
a natural-language question ("dump the local password database") shares no
tokens with the write-up's wording ("SAM hive") — that's where embeddings win.
The two retrievers fail in opposite directions, so I fuse them with **Reciprocal
Rank Fusion** (combines *rankings*, not incomparable raw scores; parameter-light
and robust) and add **MMR + a per-machine cap** so a broad "cheatsheet" query
returns coverage across many boxes instead of eight chunks from one. Broad
questions also get **query expansion** into per-technique sub-queries, because
no single dense vector sits near all technique families at once. Query-side
stopword filtering keeps the few high-value jargon tokens from being drowned by
filler words in a full-sentence question.

The ablation (see `evaluation.md`) confirms the design: BM25 is the workhorse on
this corpus, embeddings are the safety net for paraphrase, RRF+MMR trades a
little precision for coverage, and expansion buys it back on broad queries.

## One week more — the highest-leverage improvement

**Swap the stand-in embedder for a real (ideally security-tuned) model and add a
cross-encoder re-ranker.** The current evaluation ran on an offline TF-IDF+SVD
embedder (no network to fetch weights), which caps the semantic half. Deploying
`sentence-transformers/all-MiniLM-L6-v2` (or fine-tuning one on HTB/security
text), then re-ranking the fused top-30 with a cross-encoder
(`ms-marco-MiniLM-L6-v2`), would sharpen exactly the two weak questions
(JuicyPotato, Log4Shell) where the stand-in pulls loose matches. It's a
self-contained change — the backend is already pluggable — and the ablation
predicts it lifts precision on the semantic/broad queries without touching the
BM25 path that already works.
