"""
Synthesis (LLM answer generation).

Takes the retrieved chunks and asks an LLM to write a grounded, cited answer.
The whole point of the system is that the answer comes from the *retrieved HTB
write-ups*, not the model's parametric knowledge, so the system prompt is
strict:

  * Use ONLY the provided context.
  * If the context does not support an answer, say
    "Information not found in context." (verbatim).
  * Every technique mentioned MUST carry machine attribution in the exact
    format `(seen on: MachineName)`.

Providers: Anthropic (Claude) and OpenAI, selected via config.  When no API key
is available the module still returns a useful **extractive** answer (grouped,
cited snippets) so the pipeline is demonstrable end-to-end without a key.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config
from .retrieval import RetrievedChunk

logger = logging.getLogger("htb_rag.synthesis")


# --------------------------------------------------------------------------- #
# System prompt (the grounding contract)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are the HTB Cheatsheet Assistant, an offensive-security knowledge assistant.
You answer questions about penetration-testing techniques STRICTLY from a set of
retrieved Hack The Box write-up excerpts provided to you as CONTEXT.

Hard rules — follow every one:

1. GROUNDING. Use ONLY information present in the CONTEXT below. Do not use prior
   knowledge, do not guess, and do not fill gaps from memory. If the CONTEXT does
   not contain enough information to answer, reply with exactly:
   "Information not found in context."
   You may still answer the parts that ARE supported and flag the rest as not found.

2. CITATIONS (mandatory). Every technique, tool, or claim you state MUST be
   attributed to the machine(s) it came from, in the EXACT format:
       (seen on: MachineName)
   If several machines demonstrate the same technique, list them:
       (seen on: Absolute, APT)
   The machine name for each excerpt is given in its "Machine:" tag. Never invent
   a machine name; only cite machines that appear in the CONTEXT.

3. NO FABRICATION. Do not describe commands, CVEs, or tools that are not in the
   CONTEXT. If you are tempted to add a well-known detail that is not in the
   excerpts, omit it instead.

4. FORMAT. For "cheatsheet"-style questions, group findings by technique family
   with a short (1-3 sentence) explanation of each, each followed by its
   (seen on: ...) citation. For specific questions, answer directly and cite.
   Be concise and technical; this is for practitioners.
"""

USER_TEMPLATE = """\
QUESTION:
{question}

CONTEXT (retrieved HTB write-up excerpts; cite machines by their "Machine:" tag):
{context}

Write the grounded, cited answer now. Remember: only use the context above, and
attach (seen on: MachineName) to every technique.
"""


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def build_context_block(results: List[RetrievedChunk], max_chars: int = 12000) -> str:
    """
    Render retrieved chunks into a numbered context block.

    Each excerpt is explicitly labelled with its machine and section so the LLM
    can cite correctly.  We budget by characters to stay within the model window.
    """
    parts: List[str] = []
    used = 0
    for i, rc in enumerate(results, start=1):
        c = rc.chunk
        header = f"[{i}] Machine: {c.machine_name} | Section: {c.section_title}"
        body = c.raw_text.strip()
        block = f"{header}\n{body}\n"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# LLM client abstraction
# --------------------------------------------------------------------------- #
class _LLMClient:
    def complete(self, system: str, user: str, max_tokens: int,
                 temperature: float) -> str:
        raise NotImplementedError


class _AnthropicClient(_LLMClient):
    def __init__(self, cfg: Config):
        import anthropic  # lazy

        self.model = cfg.anthropic_model
        self.client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    def complete(self, system, user, max_tokens, temperature):
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class _OpenAIClient(_LLMClient):
    """Client for OpenAI and any OpenAI-compatible endpoint (Groq, Ollama, …)."""

    def __init__(self, cfg: Config, api_key: str = None,
                 base_url: str = None, model: str = None):
        from openai import OpenAI  # lazy

        self.model = model or cfg.openai_model
        # base_url=None makes the SDK use the real OpenAI endpoint.
        self.client = OpenAI(
            api_key=api_key or cfg.openai_api_key,
            base_url=base_url if base_url is not None else cfg.openai_base_url,
        )

    def complete(self, system, user, max_tokens, temperature):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class _GeminiClient(_OpenAIClient):
    """Google Gemini via its OpenAI-compatible endpoint (uses the openai SDK)."""

    def __init__(self, cfg: Config):
        super().__init__(
            cfg,
            api_key=cfg.gemini_api_key,
            base_url=cfg.gemini_base_url,
            model=cfg.gemini_model,
        )


def _get_llm_client(cfg: Config) -> Optional[_LLMClient]:
    """Instantiate the configured provider, or None if unavailable."""
    provider = cfg.llm_provider.lower()
    try:
        if provider == "anthropic" and cfg.anthropic_api_key:
            return _AnthropicClient(cfg)
        if provider == "openai" and cfg.openai_api_key:
            return _OpenAIClient(cfg)
        if provider == "gemini" and cfg.gemini_api_key:
            return _GeminiClient(cfg)
    except Exception as exc:
        logger.warning("Could not initialise %s client: %s", provider, exc)
    return None


# --------------------------------------------------------------------------- #
# Extractive fallback (no API key required)
# --------------------------------------------------------------------------- #
def _extractive_answer(question: str, results: List[RetrievedChunk]) -> str:
    """
    Deterministic, no-LLM answer: group retrieved excerpts by machine/section
    and emit a cited digest.  Not a substitute for real synthesis, but it proves
    the retrieval half and always cites correctly.
    """
    if not results:
        return "Information not found in context."

    lines = [
        "(No LLM provider configured — showing an extractive, citation-preserving "
        "digest of the retrieved context. Set GEMINI_API_KEY (--provider gemini), "
        "ANTHROPIC_API_KEY, or OPENAI_API_KEY for full synthesis.)",
        "",
        f"Question: {question}",
        "",
        "Relevant techniques found in the corpus:",
        "",
    ]
    for rc in results:
        c = rc.chunk
        # First 1-2 sentences of the excerpt as a teaser.
        snippet = re.sub(r"\s+", " ", c.raw_text).strip()
        snippet = snippet[:280] + ("..." if len(snippet) > 280 else "")
        lines.append(f"- {c.section_title} (seen on: {c.machine_name})")
        lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
@dataclass
class SynthesisResult:
    question: str
    answer: str
    used_llm: bool
    provider: str
    citations: List[str] = field(default_factory=list)   # machine names cited
    sources: List[str] = field(default_factory=list)     # source_path list
    context_chunk_ids: List[str] = field(default_factory=list)


_CITATION_RE = re.compile(r"\(seen on:\s*([^)]+)\)")


def _extract_citations(answer: str) -> List[str]:
    names: List[str] = []
    for m in _CITATION_RE.finditer(answer):
        for part in m.group(1).split(","):
            n = part.strip()
            if n and n not in names:
                names.append(n)
    return names


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def synthesize(question: str, results: List[RetrievedChunk],
               cfg: Config) -> SynthesisResult:
    """Generate the final grounded answer from retrieved chunks."""
    context = build_context_block(results)
    client = _get_llm_client(cfg)

    if client is None:
        answer = _extractive_answer(question, results)
        used_llm, provider = False, "none"
    else:
        user = USER_TEMPLATE.format(question=question, context=context)
        try:
            answer = client.complete(
                system=SYSTEM_PROMPT, user=user,
                max_tokens=cfg.llm_max_tokens, temperature=cfg.llm_temperature)
            used_llm, provider = True, cfg.llm_provider
        except Exception as exc:
            logger.error("LLM call failed (%s); falling back to extractive.", exc)
            answer = _extractive_answer(question, results)
            used_llm, provider = False, "none"

    return SynthesisResult(
        question=question,
        answer=answer,
        used_llm=used_llm,
        provider=provider,
        citations=_extract_citations(answer),
        sources=sorted({rc.chunk.source_path for rc in results}),
        context_chunk_ids=[rc.chunk.chunk_id for rc in results],
    )
