"""
End-to-end query pipeline: expand -> retrieve -> synthesize.

This is the thin glue the CLI and the evaluator both call, so the exact same
code path is exercised in production and in evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .indexing import HybridIndex
from .query_expansion import expand_query, looks_broad
from .retrieval import Retriever, RetrievedChunk
from .synthesis import SynthesisResult, synthesize


@dataclass
class QueryOutcome:
    question: str
    expanded_queries: List[str]
    results: List[RetrievedChunk]
    synthesis: Optional[SynthesisResult]


class RAGPipeline:
    def __init__(self, index: HybridIndex, use_llm_expansion: bool = False):
        self.index = index
        self.cfg = index.cfg
        self.retriever = Retriever(index)
        self.use_llm_expansion = use_llm_expansion

    # ---- retrieval only (used heavily by the evaluator) ------------------
    def retrieve(self, question: str, top_k: Optional[int] = None,
                 expand: bool = True) -> List[RetrievedChunk]:
        if expand and looks_broad(question):
            subs = expand_query(question, self.cfg, use_llm=self.use_llm_expansion)
            return self.retriever.retrieve_multi(subs, top_k=top_k)
        return self.retriever.retrieve(question, top_k=top_k)

    # ---- full pipeline ---------------------------------------------------
    def answer(self, question: str, top_k: Optional[int] = None,
               expand: bool = True, synthesize_answer: bool = True) -> QueryOutcome:
        if expand and looks_broad(question):
            subs = expand_query(question, self.cfg, use_llm=self.use_llm_expansion)
            results = self.retriever.retrieve_multi(subs, top_k=top_k)
        else:
            subs = [question]
            results = self.retriever.retrieve(question, top_k=top_k)

        synth = synthesize(question, results, self.cfg) if synthesize_answer else None
        return QueryOutcome(
            question=question,
            expanded_queries=subs,
            results=results,
            synthesis=synth,
        )
