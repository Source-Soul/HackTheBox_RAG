"""
Evaluation harness.

Scores the retriever against a hand-built test set (data/testset.json).  Each
test item carries a set of *gold machines* — the machines that a human verified
(by grepping/reading the raw files) as demonstrating the technique in question.

We report retrieval quality at the **machine** granularity, which is what the
assessment's citation requirement cares about ("which machine(s) demonstrate
it"):

    recall@k    = |gold ∩ retrieved_machines| / |gold|
    precision@k = |gold ∩ retrieved_machines| / |retrieved_machines|
    hit@k       = 1 if at least one gold machine retrieved else 0
    MRR         = 1 / rank of first gold-machine chunk

Machine-level precision is intentionally lenient about *extra* machines that
are relevant-but-not-in-gold (the gold set is a human sample, not exhaustive),
so treat precision as a lower bound and read it alongside recall and hit-rate.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .ingestion import normalize_machine_slug
from .pipeline import RAGPipeline


@dataclass
class QuestionScore:
    id: str
    question: str
    gold_machines: List[str]
    retrieved_machines: List[str]
    recall: float
    precision: float
    hit: int
    mrr: float
    f1: float


@dataclass
class EvalReport:
    k: int
    per_question: List[QuestionScore] = field(default_factory=list)
    macro_recall: float = 0.0
    macro_precision: float = 0.0
    macro_f1: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _slugset(names: List[str]) -> set:
    return {normalize_machine_slug(n) for n in names}


def evaluate(pipeline: RAGPipeline, testset_path: Path, k: int = 8,
             expand: bool = True) -> EvalReport:
    items = json.loads(Path(testset_path).read_text())
    scores: List[QuestionScore] = []

    for item in items:
        q = item["question"]
        gold = item.get("gold_machines", [])
        gold_slugs = _slugset(gold)

        results = pipeline.retrieve(q, top_k=k, expand=expand)

        # Ordered, de-duplicated list of retrieved machines (for MRR + precision).
        retrieved_machines: List[str] = []
        retrieved_slugs_ordered: List[str] = []
        for rc in results:
            slug = rc.chunk.machine_slug
            if slug not in retrieved_slugs_ordered:
                retrieved_slugs_ordered.append(slug)
                retrieved_machines.append(rc.chunk.machine_name)

        retrieved_set = set(retrieved_slugs_ordered)
        inter = gold_slugs & retrieved_set

        recall = len(inter) / len(gold_slugs) if gold_slugs else 0.0
        precision = len(inter) / len(retrieved_set) if retrieved_set else 0.0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
        hit = 1 if inter else 0

        # MRR over the chunk ranking (first chunk whose machine is gold).
        mrr = 0.0
        for rank, rc in enumerate(results, start=1):
            if rc.chunk.machine_slug in gold_slugs:
                mrr = 1.0 / rank
                break

        scores.append(QuestionScore(
            id=item.get("id", ""), question=q, gold_machines=gold,
            retrieved_machines=retrieved_machines, recall=recall,
            precision=precision, hit=hit, mrr=mrr, f1=f1))

    report = EvalReport(k=k, per_question=scores)
    if scores:
        report.macro_recall = statistics.mean(s.recall for s in scores)
        report.macro_precision = statistics.mean(s.precision for s in scores)
        report.macro_f1 = statistics.mean(s.f1 for s in scores)
        report.hit_rate = statistics.mean(s.hit for s in scores)
        report.mrr = statistics.mean(s.mrr for s in scores)
    return report


def format_report(report: EvalReport) -> str:
    lines = []
    lines.append(f"{'ID':<6}{'recall':>8}{'prec':>8}{'f1':>8}{'hit':>5}{'mrr':>7}  question")
    lines.append("-" * 92)
    for s in report.per_question:
        lines.append(
            f"{s.id:<6}{s.recall:>8.2f}{s.precision:>8.2f}{s.f1:>8.2f}"
            f"{s.hit:>5}{s.mrr:>7.2f}  {s.question[:44]}")
    lines.append("-" * 92)
    lines.append(
        f"{'MACRO':<6}{report.macro_recall:>8.2f}{report.macro_precision:>8.2f}"
        f"{report.macro_f1:>8.2f}{report.hit_rate:>5.2f}{report.mrr:>7.2f}"
        f"  (k={report.k}, n={len(report.per_question)})")
    return "\n".join(lines)
