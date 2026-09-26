"""
Ingestion & parsing.

Turns raw Markdown write-ups into structured `Section` records:

    {machine_name, section_title, content, source_path, stage, heading_level}

Design priorities (the corner cases the spec calls out):

1. **Code-fence awareness.**  HTB write-ups are full of shell transcripts that
   contain lines beginning with `#` (root prompts, bash comments).  Naively
   treating every `#`-line as a heading corrupts ~2,800 false sections across
   this corpus.  We track ``` / ~~~ fence state and only split on *real* ATX
   headings that live outside code blocks.

2. **Missing / non-standard headers.**  Some content is plain prose before the
   first heading, or uses headings like "PrivEsc", "Root", "Beyond Root",
   "Shell as SYSTEM" instead of a canonical scheme.  We (a) capture any preamble
   as a synthetic "Overview" section, and (b) classify every heading into a
   logical *stage* (recon / web / foothold / privesc / other) so downstream
   code can reason about privilege-escalation content regardless of wording.

3. **Inconsistent machine naming.**  The machine name is resolved with a
   multi-strategy fallback (title line -> image alt text -> filename slug) and
   normalised so that "htb-admirer_two.md", "admirer two" and "Admirer-Two"
   all collapse to a single canonical name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional

from .config import (
    Config,
    SECTION_STAGE_PATTERNS,
    BOILERPLATE_HEADINGS,
)

# --------------------------------------------------------------------------- #
# Regexes (compiled once)
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")          # opening/closing code fence
_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")  # markdown ATX heading
_TITLE_LINK_RE = re.compile(r"^\[HTB:\s*([^\]]+)\]", re.IGNORECASE)  # [HTB: Name]
_IMG_ALT_RE = re.compile(r"^!\[([^\]]+)\]")          # ![Name-cover](...)
_STAGE_RES = [(stage, re.compile(pat, re.IGNORECASE)) for stage, pat in SECTION_STAGE_PATTERNS]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Section:
    """A structured, header-scoped slice of one write-up."""
    machine_name: str          # canonical, human-readable, e.g. "Admirer Two"
    machine_slug: str          # normalised key, e.g. "admirertwo"
    section_title: str         # raw heading text, e.g. "Shell as SYSTEM"
    content: str               # section body (markdown, incl. code blocks)
    source_path: str           # relative path to the .md file
    stage: str                 # logical stage: recon/web/foothold/privesc/other
    heading_level: int         # 1-6, or 0 for synthetic preamble sections
    heading_path: List[str] = field(default_factory=list)  # breadcrumb of parents

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Machine-name resolution
# --------------------------------------------------------------------------- #
def normalize_machine_slug(raw: str) -> str:
    """
    Collapse inconsistent naming into a single comparable key.

    "htb-admirer_two.md" -> "admirertwo"
    "Admirer Two"        -> "admirertwo"
    "APT"                -> "apt"
    """
    s = raw.strip().lower()
    s = re.sub(r"\.md$", "", s)
    s = re.sub(r"^htb[-_ ]+", "", s)          # drop leading htb- prefix
    s = re.sub(r"[^a-z0-9]+", "", s)          # drop spaces/underscores/hyphens
    return s


def _prettify_slug(slug_source: str) -> str:
    """Turn a filename slug into a readable name as a last-resort fallback."""
    base = re.sub(r"\.md$", "", slug_source)
    base = re.sub(r"^htb[-_ ]+", "", base, flags=re.IGNORECASE)
    parts = re.split(r"[-_\s]+", base)
    return " ".join(p.capitalize() for p in parts if p)


def resolve_machine_name(path: Path, text: str) -> str:
    """
    Multi-strategy, best-effort resolution of the human-readable machine name.

    Order of preference:
      1. Title link on the first non-empty line:  [HTB: Absolute](...)
      2. First image alt text:  ![Absolute-cover](...)  -> strip -cover/-banner
      3. Filename slug, title-cased.
    """
    # Look at the first few non-empty lines only (name always lives at the top).
    head_lines = [ln.strip() for ln in text.splitlines()[:15] if ln.strip()]

    for ln in head_lines:
        m = _TITLE_LINK_RE.match(ln)
        if m:
            return m.group(1).strip()

    for ln in head_lines:
        m = _IMG_ALT_RE.match(ln)
        if m:
            alt = m.group(1).strip()
            # Alt text is often "Name-cover" / "Name-banner" / "Name cover".
            alt = re.sub(r"[-_\s]+(cover|banner|logo|card)$", "", alt, flags=re.IGNORECASE)
            if alt:
                return alt.strip()

    # Fallback: derive from filename.
    return _prettify_slug(path.name)


# --------------------------------------------------------------------------- #
# Stage classification
# --------------------------------------------------------------------------- #
def classify_stage(heading_text: str) -> str:
    """Map a raw heading onto a logical stage using the ordered pattern table."""
    for stage, rx in _STAGE_RES:
        if rx.search(heading_text):
            return stage
    return "other"


def _unescape_heading(text: str) -> str:
    """Strip backslash escapes markdown authors use in headings (\\_ \\[ \\* ...)."""
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", text).strip()


# --------------------------------------------------------------------------- #
# Core parser
# --------------------------------------------------------------------------- #
def _iter_heading_blocks(text: str):
    """
    Yield (heading_level, heading_text, body_lines) tuples for a document,
    ignoring any `#` lines that occur *inside* fenced code blocks.

    A synthetic block with level 0 and title None is yielded first if the file
    opens with prose before its first real heading (the "missing header" case).
    """
    lines = text.splitlines()
    in_fence = False
    fence_marker = ""

    current_level: Optional[int] = None
    current_title: Optional[str] = None
    buffer: List[str] = []

    def flush():
        if current_title is not None or any(l.strip() for l in buffer):
            yield_level = current_level if current_level is not None else 0
            return (yield_level, current_title, list(buffer))
        return None

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0] * 3  # normalise to ``` or ~~~
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif line.strip().startswith(fence_marker):
                in_fence = False
            buffer.append(line)
            continue

        if not in_fence:
            m = _ATX_RE.match(line)
            if m:
                # Close the previous block before starting a new one.
                flushed = flush()
                if flushed:
                    yield flushed
                current_level = len(m.group(1))
                current_title = m.group(2).strip()
                buffer = []
                continue

        buffer.append(line)

    flushed = flush()
    if flushed:
        yield flushed


def parse_markdown(path: Path, corpus_dir: Path) -> List[Section]:
    """
    Parse a single write-up into a list of `Section` records.

    Guarantees:
      * every section carries the machine name (metadata attached at the source),
      * headings inside code fences are never treated as section boundaries,
      * prose before the first heading is preserved as an "Overview" section,
      * a file with *no* headings at all still yields exactly one section.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    machine_name = resolve_machine_name(path, text)
    machine_slug = normalize_machine_slug(path.name)
    try:
        source_path = str(path.relative_to(corpus_dir))
    except ValueError:
        source_path = str(path)

    sections: List[Section] = []
    # Breadcrumb stack of (level, title, stage) to build a heading_path and to
    # inherit the logical stage from ancestor headings.
    heading_stack: List[tuple] = []

    for level, title, body_lines in _iter_heading_blocks(text):
        content = "\n".join(body_lines).strip()

        if title is None:
            # Preamble / prose with no heading -> synthetic Overview section.
            title_effective = "Overview"
            stage = "recon"
            level_effective = 0
            heading_path: List[str] = []
        else:
            title_effective = _unescape_heading(title)
            # Maintain the breadcrumb stack.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_path = [t for _, t, _ in heading_stack]
            own_stage = classify_stage(title_effective)
            # Stage inheritance: a generic sub-heading ("RunAsCs", "Compile")
            # inherits the nearest meaningful ancestor stage (e.g. its parent
            # "Shell as Administrator" -> privesc) so privesc content stays
            # tagged even when the leaf heading is uninformative.
            if own_stage == "other":
                for _, _, parent_stage in reversed(heading_stack):
                    if parent_stage != "other":
                        own_stage = parent_stage
                        break
            stage = own_stage
            heading_stack.append((level, title_effective, stage))
            level_effective = level

        # Drop pure boilerplate (e.g. "Box Info") that has no offensive content.
        if title_effective.strip().lower() in BOILERPLATE_HEADINGS:
            continue
        # Skip empty sections (a heading immediately followed by a sub-heading).
        if not content:
            continue

        sections.append(
            Section(
                machine_name=machine_name,
                machine_slug=machine_slug,
                section_title=title_effective,
                content=content,
                source_path=source_path,
                stage=stage,
                heading_level=level_effective,
                heading_path=heading_path,
            )
        )

    # Absolute fallback: a file that produced nothing (e.g. only a Box Info
    # block) still deserves one record so it is retrievable.
    if not sections:
        body = text.strip()
        if body:
            sections.append(
                Section(
                    machine_name=machine_name,
                    machine_slug=machine_slug,
                    section_title="Overview",
                    content=body,
                    source_path=source_path,
                    stage="other",
                    heading_level=0,
                    heading_path=[],
                )
            )
    return sections


def ingest_corpus(cfg: Config) -> List[Section]:
    """Parse every ``*.md`` file under ``cfg.corpus_dir`` into Section records."""
    corpus_dir = cfg.corpus_dir
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}. "
            "Clone https://github.com/0xh7ml/htb-wiki and point "
            "cfg.corpus_dir at its 'raw' folder."
        )

    md_files = sorted(corpus_dir.rglob("*.md"))
    all_sections: List[Section] = []
    for path in md_files:
        all_sections.extend(parse_markdown(path, corpus_dir))
    return all_sections


# --------------------------------------------------------------------------- #
# Manual smoke test:  python -m htb_rag.ingestion htb-wiki/raw/htb-absolute.md
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m htb_rag.ingestion <file.md>")
        raise SystemExit(1)

    p = Path(sys.argv[1])
    secs = parse_markdown(p, p.parent)
    print(f"machine = {secs[0].machine_name!r}  slug = {secs[0].machine_slug!r}")
    print(f"{len(secs)} sections parsed:\n")
    for s in secs:
        print(f"  [{s.stage:8}] L{s.heading_level} {s.section_title!r} "
              f"({len(s.content)} chars)")
