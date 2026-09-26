"""
Section-aware chunking.

Strategy
--------
1. **Section-aware, not blind sliding window.**  Each `Section` from ingestion
   is a self-contained technique unit ("Shell as SYSTEM", "ADCS abuse", ...).
   We keep a section whole whenever it fits in the token budget, so a complete
   vulnerability/technique unit is never split mid-explanation.

2. **Sub-chunk only when necessary.**  Sections longer than the budget are
   split into ~300-500 token windows with a 50-token overlap.  The split is
   *fence-aware*: we never cut in the middle of a fenced code block, because a
   half a shell transcript is useless (and misleads the embedder / BM25).

3. **Metadata on every chunk (the crucial rule).**  Every emitted chunk carries
   `machine_name` and `section_title` (plus stage, source path and a heading
   breadcrumb).  We also prepend a compact human-readable header to the chunk
   *text* itself ("Machine: X | Section: Y"), which measurably improves both
   lexical matching (the machine name becomes a searchable token) and the
   grounding of the synthesis step.

Token counting is pluggable: we use `tiktoken` when available, else a
deterministic word-based heuristic so the module has no hard dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

from .config import Config, BOILERPLATE_HEADINGS
from .ingestion import Section

# --------------------------------------------------------------------------- #
# Token counting (optional tiktoken, graceful fallback)
# --------------------------------------------------------------------------- #
_TOKENIZER: Optional[Callable[[str], int]] = None


def _get_token_counter() -> Callable[[str], int]:
    """Return a `text -> token_count` function, memoised."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

        _TOKENIZER = _count
    except Exception:
        # Heuristic: ~0.75 words per token for English + code.  We count
        # whitespace-separated words and scale.  Deterministic and dependency
        # free; good enough for *sizing* chunks (not for billing).
        def _count(text: str) -> int:
            words = re.findall(r"\S+", text)
            return int(round(len(words) / 0.75))

        _TOKENIZER = _count

    return _TOKENIZER


def count_tokens(text: str) -> int:
    return _get_token_counter()(text)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    chunk_id: str              # stable id: "<slug>::<section#>::<part#>"
    machine_name: str
    machine_slug: str
    section_title: str
    stage: str
    source_path: str
    text: str                  # chunk body WITH the metadata header prepended
    raw_text: str              # chunk body WITHOUT the header (for display)
    token_count: int
    heading_path: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


# --------------------------------------------------------------------------- #
# Fence-aware line grouping
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


def _split_into_atoms(content: str) -> List[str]:
    """
    Break a section body into indivisible 'atoms'.

    A fenced code block is one atom (never split internally).  Runs of prose
    between fences are further split into paragraph atoms on blank lines, so the
    packer can place window boundaries at natural gaps.
    """
    lines = content.splitlines()
    atoms: List[str] = []
    buf: List[str] = []
    in_fence = False
    fence_marker = ""

    def flush_prose():
        # Split accumulated prose on blank lines into paragraph atoms.
        block = "\n".join(buf).strip("\n")
        if block.strip():
            for para in re.split(r"\n\s*\n", block):
                if para.strip():
                    atoms.append(para.strip("\n"))
        buf.clear()

    for line in lines:
        fm = _FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)[0] * 3
            if not in_fence:
                flush_prose()
                in_fence, fence_marker = True, marker
                buf.append(line)
            elif line.strip().startswith(fence_marker):
                buf.append(line)
                atoms.append("\n".join(buf))  # whole code block = one atom
                buf.clear()
                in_fence = False
            else:
                buf.append(line)
            continue
        buf.append(line)

    # Trailing content.
    if in_fence:
        atoms.append("\n".join(buf))  # unterminated fence: keep as one atom
    else:
        flush_prose()
    return [a for a in atoms if a.strip()]


# --------------------------------------------------------------------------- #
# Overlap helper
# --------------------------------------------------------------------------- #
def _tail_overlap(text: str, overlap_tokens: int) -> str:
    """Return the trailing ~overlap_tokens worth of text (word-granular)."""
    if overlap_tokens <= 0:
        return ""
    words = re.findall(r"\S+\s*", text)
    # Approx words for the requested token budget.
    n_words = max(1, int(overlap_tokens * 0.75))
    return "".join(words[-n_words:]).strip()


# --------------------------------------------------------------------------- #
# Core chunker
# --------------------------------------------------------------------------- #
def _header_for(section: Section) -> str:
    """Compact metadata header prepended to every chunk's searchable text."""
    path = [h for h in section.heading_path
            if h.strip().lower() not in BOILERPLATE_HEADINGS]
    crumbs = " > ".join(path) if path else ""
    loc = f"{section.section_title}"
    if crumbs:
        loc = f"{crumbs} > {section.section_title}"
    return f"[Machine: {section.machine_name} | Section: {loc}]"


def chunk_section(section: Section, cfg: Config) -> List[Chunk]:
    """Turn one Section into one or more Chunks, honouring the token budget."""
    header = _header_for(section)
    header_tokens = count_tokens(header)
    body_budget = max(cfg.chunk_min_tokens, cfg.chunk_max_tokens - header_tokens)
    target = max(cfg.chunk_min_tokens, cfg.chunk_target_tokens - header_tokens)

    total_tokens = count_tokens(section.content)

    # ---- Fast path: whole section fits -> keep the technique unit intact ----
    if total_tokens <= body_budget:
        return [_make_chunk(section, header, section.content, part=0)]

    # ---- Slow path: pack atoms into overlapping windows --------------------
    atoms = _split_into_atoms(section.content)
    chunks: List[Chunk] = []
    window: List[str] = []
    window_tokens = 0
    part = 0
    carry_overlap = ""  # overlap text carried from the previous window

    def emit(force: bool = False):
        nonlocal window, window_tokens, part, carry_overlap
        if not window:
            return
        body = "\n\n".join(window).strip()
        if not body:
            window, window_tokens = [], 0
            return
        chunks.append(_make_chunk(section, header, body, part=part))
        part += 1
        # Seed the next window with a token-bounded overlap for continuity.
        carry_overlap = _tail_overlap(body, cfg.chunk_overlap_tokens)
        window = []
        window_tokens = 0

    for atom in atoms:
        atom_tokens = count_tokens(atom)

        # A single atom (usually a long code block) bigger than the budget:
        # emit whatever is buffered, then hard-split the atom by lines.
        if atom_tokens > body_budget:
            emit(force=True)
            for piece in _hard_split_atom(atom, body_budget, cfg.chunk_overlap_tokens):
                chunks.append(_make_chunk(section, header, piece, part=part))
                part += 1
            carry_overlap = _tail_overlap(atom, cfg.chunk_overlap_tokens)
            continue

        # Start a fresh window with carried overlap for context continuity.
        if not window and carry_overlap:
            window.append(carry_overlap)
            window_tokens += count_tokens(carry_overlap)
            carry_overlap = ""

        # If adding this atom would blow the ceiling, close the window first.
        if window and window_tokens + atom_tokens > body_budget:
            emit()
            if carry_overlap:
                window.append(carry_overlap)
                window_tokens += count_tokens(carry_overlap)
                carry_overlap = ""

        window.append(atom)
        window_tokens += atom_tokens

        # Prefer to close near the target size at a natural atom boundary.
        if window_tokens >= target:
            emit()

    emit(force=True)

    # Merge a tiny trailing fragment back into its predecessor.
    if len(chunks) >= 2 and chunks[-1].token_count < cfg.chunk_min_tokens:
        tail = chunks.pop()
        prev = chunks.pop()
        merged_body = prev.raw_text + "\n\n" + tail.raw_text
        chunks.append(_make_chunk(section, header, merged_body, part=prev_part(prev)))
    return chunks


def prev_part(chunk: Chunk) -> int:
    return int(chunk.chunk_id.rsplit("::", 1)[-1])


def _hard_split_atom(atom: str, budget: int, overlap: int) -> List[str]:
    """Split an over-long atom (e.g. a huge code block) by lines with overlap."""
    lines = atom.splitlines()
    pieces: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for line in lines:
        lt = count_tokens(line)
        if cur and cur_tokens + lt > budget:
            pieces.append("\n".join(cur))
            # carry a few trailing lines as overlap
            keep = max(1, int(len(cur) * (overlap / max(1, cur_tokens))))
            cur = cur[-keep:]
            cur_tokens = count_tokens("\n".join(cur))
        cur.append(line)
        cur_tokens += lt
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _make_chunk(section: Section, header: str, body: str, part: int) -> Chunk:
    raw = body.strip()
    text = f"{header}\n{raw}"
    # A stable, readable id.  Section index is not known here, so we use a hash
    # of the source + title + part; collisions are astronomically unlikely.
    import hashlib

    sig = f"{section.source_path}|{section.section_title}|{part}"
    short = hashlib.sha1(sig.encode()).hexdigest()[:8]
    chunk_id = f"{section.machine_slug}::{short}::{part}"
    return Chunk(
        chunk_id=chunk_id,
        machine_name=section.machine_name,
        machine_slug=section.machine_slug,
        section_title=section.section_title,
        stage=section.stage,
        source_path=section.source_path,
        text=text,
        raw_text=raw,
        token_count=count_tokens(text),
        heading_path=list(section.heading_path),
    )


def chunk_sections(sections: List[Section], cfg: Config) -> List[Chunk]:
    """Chunk a list of sections into the flat chunk list used for indexing."""
    chunks: List[Chunk] = []
    for sec in sections:
        chunks.extend(chunk_section(sec, cfg))
    return chunks


# --------------------------------------------------------------------------- #
# Manual smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from .ingestion import parse_markdown

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("htb-wiki/raw/htb-absolute.md")
    cfg = Config()
    secs = parse_markdown(p, p.parent)
    chunks = chunk_sections(secs, cfg)
    print(f"{len(secs)} sections -> {len(chunks)} chunks")
    sizes = [c.token_count for c in chunks]
    print(f"token sizes: min={min(sizes)} max={max(sizes)} "
          f"mean={sum(sizes)//len(sizes)}")
    for c in chunks[:4]:
        print("-" * 70)
        print(c.chunk_id, "|", c.token_count, "tok")
        print(c.text[:300])
