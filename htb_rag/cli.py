"""
Command-line interface for the HTB Cheatsheet RAG system.

Subcommands
-----------
    index    Build (and persist) the hybrid index from the raw corpus.
    query    Ask a question; retrieve + synthesize a grounded, cited answer.
    eval     Score retrieval against data/testset.json.
    info     Print metadata about the built index.

Examples
--------
    python -m htb_rag.cli index  --corpus htb-wiki/raw --index-dir index_store
    python -m htb_rag.cli query  "Provide me the Windows privilege escalation cheatsheet."
    python -m htb_rag.cli query  "How does JuicyPotato work and which machines use it?" --k 6
    python -m htb_rag.cli eval   --testset data/testset.json --k 8
    python -m htb_rag.cli info
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .pipeline import RAGPipeline


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _load_config(args) -> Config:
    overrides = {}
    if getattr(args, "corpus", None):
        overrides["corpus_dir"] = args.corpus
    if getattr(args, "index_dir", None):
        overrides["index_dir"] = args.index_dir
    if getattr(args, "provider", None):
        overrides["llm_provider"] = args.provider
    return Config.from_env(**overrides)


# --------------------------------------------------------------------------- #
# Subcommand: index
# --------------------------------------------------------------------------- #
def cmd_index(args):
    from .indexing import build_index

    cfg = _load_config(args)
    print(f"[*] Building index from {cfg.corpus_dir} -> {cfg.index_dir}")
    index = build_index(cfg, prefer_offline=args.offline, show_progress=not args.quiet)
    index.save()
    print(f"[+] Done. {len(index.chunks)} chunks indexed.")
    print(f"    lexical={index.bm25_backend}  "
          f"embedder={getattr(index.embedder, 'name', '?')}  "
          f"vector={getattr(index.vector_index, 'name', '?')}")


# --------------------------------------------------------------------------- #
# Subcommand: query
# --------------------------------------------------------------------------- #
def cmd_query(args):
    from .indexing import load_index

    cfg = _load_config(args)
    index = load_index(cfg)
    pipe = RAGPipeline(index, use_llm_expansion=args.llm_expansion)

    outcome = pipe.answer(
        args.question, top_k=args.k, expand=not args.no_expand,
        synthesize_answer=not args.retrieval_only)

    if args.no_expand is False and len(outcome.expanded_queries) > 1:
        print(f"[*] Expanded into {len(outcome.expanded_queries)} sub-queries.")

    print("\n=== RETRIEVED CONTEXT " + "=" * 48)
    for i, rc in enumerate(outcome.results, 1):
        print(f"[{i}] {rc.chunk.machine_name} | {rc.chunk.section_title} "
              f"(stage={rc.chunk.stage}, bm25#={rc.bm25_rank}, vec#={rc.vector_rank})")
        if args.show_snippets:
            snippet = rc.chunk.raw_text.strip().replace("\n", " ")[:200]
            print(f"      {snippet}...")

    if outcome.synthesis is not None:
        s = outcome.synthesis
        print("\n=== ANSWER " + "=" * 59)
        print(s.answer)
        print("\n=== METADATA " + "=" * 57)
        print(f"provider={s.provider}  used_llm={s.used_llm}")
        print(f"machines cited: {', '.join(s.citations) if s.citations else '(none)'}")
        print(f"sources: {', '.join(s.sources)}")


# --------------------------------------------------------------------------- #
# Subcommand: eval
# --------------------------------------------------------------------------- #
def cmd_eval(args):
    from .indexing import load_index
    from .evaluation import evaluate, format_report
    import json

    cfg = _load_config(args)
    index = load_index(cfg)
    pipe = RAGPipeline(index, use_llm_expansion=args.llm_expansion)

    report = evaluate(pipe, Path(args.testset), k=args.k, expand=not args.no_expand)
    print(format_report(report))

    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\n[+] Wrote detailed report to {args.out}")


# --------------------------------------------------------------------------- #
# Subcommand: info
# --------------------------------------------------------------------------- #
def cmd_info(args):
    import json

    cfg = _load_config(args)
    meta_path = cfg.index_dir / "meta.json"
    if not meta_path.exists():
        print(f"No index at {cfg.index_dir}. Run `index` first.")
        return
    print(json.dumps(json.loads(meta_path.read_text()), indent=2))


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # Common flags shared by every subcommand (so they work in any position).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="INFO logging")
    common.add_argument("--corpus", help="path to raw *.md corpus (default: config)")
    common.add_argument("--index-dir", help="path to the index store (default: config)")

    p = argparse.ArgumentParser(
        prog="python -m htb_rag.cli", parents=[common],
        description="HTB Cheatsheet Assistant — hybrid RAG over HTB write-ups.")
    sub = p.add_subparsers(dest="command", required=True)

    # index
    pi = sub.add_parser("index", help="build the hybrid index", parents=[common])
    pi.add_argument("--offline", action="store_true",
                    help="force offline backends (pure-python BM25 + TF-IDF)")
    pi.add_argument("--quiet", action="store_true", help="hide embedding progress")
    pi.set_defaults(func=cmd_index)

    # query
    pq = sub.add_parser("query", help="ask a question", parents=[common])
    pq.add_argument("question", help="the natural-language question")
    pq.add_argument("--k", type=int, default=None, help="number of chunks to retrieve")
    pq.add_argument("--provider", choices=["gemini", "anthropic", "openai", "none"],
                    help="LLM provider for synthesis")
    pq.add_argument("--no-expand", action="store_true",
                    help="disable query expansion for broad questions")
    pq.add_argument("--llm-expansion", action="store_true",
                    help="use the LLM (not heuristics) to expand broad queries")
    pq.add_argument("--retrieval-only", action="store_true",
                    help="skip synthesis, show retrieved context only")
    pq.add_argument("--show-snippets", action="store_true",
                    help="print a snippet of each retrieved chunk")
    pq.set_defaults(func=cmd_query)

    # eval
    pe = sub.add_parser("eval", help="score retrieval against a test set", parents=[common])
    pe.add_argument("--testset", default="data/testset.json")
    pe.add_argument("--k", type=int, default=8)
    pe.add_argument("--no-expand", action="store_true")
    pe.add_argument("--llm-expansion", action="store_true")
    pe.add_argument("--out", help="write a detailed JSON report to this path")
    pe.set_defaults(func=cmd_eval)

    # info
    pinfo = sub.add_parser("info", help="print index metadata", parents=[common])
    pinfo.set_defaults(func=cmd_info)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
