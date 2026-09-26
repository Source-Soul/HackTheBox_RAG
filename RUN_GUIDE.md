# RUN GUIDE — HTB Cheatsheet RAG (PyCharm, with free Gemini synthesis)

Follow these once, top to bottom. Commands are run in the **PyCharm Terminal** (bottom of
the window) from the **project root** (the folder containing `htb_rag/`).

---

## 0. Prerequisites
- Python 3.9+ (built/tested on 3.11)
- PyCharm (Community is fine) + Git

## 1. Open the project
File → Open → select the unzipped `htb_rag_project` folder.

## 2. Create the virtual environment
Settings → Project → Python Interpreter → Add Interpreter → Add Local → **Virtualenv → New** →
OK. Open the Terminal tab; you should see `(.venv)` in the prompt.

## 3. Install dependencies
```bash
pip install -r requirements.txt
```
- First use downloads a ~90 MB `sentence-transformers` model (needs internet once).
- If the heavy libs won't install, you can run everything offline — add `--offline` to the
  build step in §5 and skip nothing else. `openai` (for Gemini) still installs fine on its own.

## 4. Get the dataset (corpus)
```bash
git clone https://github.com/0xh7ml/htb-wiki.git
```
This creates `htb-wiki/raw/` (513 write-ups) next to the `htb_rag/` package.

## 5. Build the index (once, ~25 s)
```bash
python -m htb_rag.cli index --corpus htb-wiki/raw -v
```
Add `--offline` if you skipped the heavy libs. Creates `index_store/`.

**Expected tail:**
```
[+] Done. 17291 chunks indexed.
    lexical=rank_bm25.BM25Okapi  embedder=sentence-transformers:...  vector=faiss:IndexFlatIP
# (offline mode: lexical=PurePythonBM25  embedder=tfidf-svd:256  vector=numpy:flat-ip)
```

## 6. Set up FREE Gemini synthesis
1. Get a key at https://aistudio.google.com → **Get API key** (no credit card).
2. Install the SDK (if not already): `pip install openai`
3. Set the key. **Do not paste it into any file** — use an environment variable:

   macOS/Linux terminal:
   ```bash
   export GEMINI_API_KEY="AIza...your_key..."
   ```
   Windows PowerShell:
   ```powershell
   $env:GEMINI_API_KEY="AIza...your_key..."
   ```
   Or in PyCharm: Run → Edit Configurations → **Environment variables** →
   `GEMINI_API_KEY=AIza...`

   Model/base-URL are already defaulted (`gemini-2.0-flash`); override with
   `GEMINI_MODEL` if you like.

## 7. Ask a question (with real synthesis)
```bash
python -m htb_rag.cli query "Provide me the Windows privilege escalation cheatsheet." --provider gemini
```
**Expected:** a `RETRIEVED CONTEXT` list, then an `ANSWER` block of grouped prose where every
technique ends with `(seen on: <Machine>)`, then:
```
provider=gemini  used_llm=True
```
Without a key it prints the extractive digest instead and `provider=none used_llm=False` —
still fully functional, just not LLM-written.

Good questions to try (they exercise the edge cases):
```bash
python -m htb_rag.cli query "Which machines exploit the XZ Utils backdoor (CVE-2024-3094)?" --provider gemini
#   -> should answer: "Information not found in context." (nothing invented)
python -m htb_rag.cli query "How do I get root using sudo and SeImpersonatePrivilege?" --provider gemini
#   -> should separate the Linux (sudo) and Windows (SeImpersonate) paths
```

## 8. Run the evaluation
```bash
python -m htb_rag.cli eval --testset data/testset.json --k 8 --out data/eval_report.json
```
**Expected:** a per-question table then:
```
MACRO     0.20    0.86    0.29 1.00   0.97  (k=8, n=16)
```
(precision 0.86, hit-rate 1.00, MRR 0.97 on the offline backend; higher with the full
sentence-transformers stack). Eval scores retrieval and is independent of the LLM provider.

## 9. Inspect the index (optional)
```bash
python -m htb_rag.cli info
```

---

## Troubleshooting
| Symptom | Fix |
|---|---|
| `No module named htb_rag` | Working directory isn't the project root (the folder with `htb_rag/`). |
| `No index found … Run index first` | You skipped §5, or built in a different working dir. |
| `Corpus directory not found` | `git clone` didn't land in the project root; `htb-wiki/raw/` must sit next to `htb_rag/`. |
| `provider=none` though key is set | Run with `-v`; common causes: `openai` SDK not installed, wrong key, or a model name your key can't access. It falls back to extractive, never crashes. |
| Gemini rate limit / 429 | Free tier is rate-limited; wait a moment or lower how many queries you run back-to-back. |

## Security note
Never commit or zip your API key. It belongs only in an environment variable. If a key was
ever shared in plaintext, rotate it in Google AI Studio.
