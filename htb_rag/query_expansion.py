"""
Query expansion for broad "cheatsheet" questions.

A single dense query vector is a poor match for a request like
"Windows privilege escalation cheatsheet": the ideal answer spans many
*distinct* technique families (SAM/SYSTEM hive dumps, potato-family exploits,
ADCS abuse, ACL/BloodHound abuse, service/registry misconfig, token abuse ...),
and no single embedding sits near all of them at once.

The fix is multi-query retrieval: decompose the broad question into several
focused sub-queries, retrieve for each, and fuse.  Two expansion strategies:

  * **LLM expansion** (preferred when an API key is configured): ask the model
    for 6-8 specific sub-queries grounded in offensive-security terminology.
  * **Heuristic expansion** (offline default): detect broad intents by keyword
    and expand from a curated seed table of technique families.  This keeps the
    headline use case working with zero external dependencies.

The original query is always retained in the expansion set.
"""

from __future__ import annotations

import re
from typing import List

from .config import Config

# --------------------------------------------------------------------------- #
# Curated technique seeds per broad intent.  These are deliberately phrased as
# retrieval queries (jargon-dense) because BM25 is doing much of the work.
# --------------------------------------------------------------------------- #
_SEEDS = {
    "windows_privesc": [
        "dump SAM and SYSTEM registry hive secretsdump local hashes",
        "JuicyPotato RoguePotato PrintSpoofer SeImpersonatePrivilege potato exploit",
        "ADCS AD CS certificate template abuse ESC1 Certipy",
        "BloodHound ACL abuse GenericWrite GenericAll WriteDACL DCSync",
        "unquoted service path weak service permissions privilege escalation",
        "SeBackupPrivilege SeRestorePrivilege token privilege abuse",
        "GPP cpassword AutoLogon stored credentials registry",
        "Windows kernel exploit MS16-032 Watson exploit suggester",
        "Kerberoasting AS-REP roasting domain user to admin",
        "DPAPI credential manager stored secrets extraction",
    ],
    "linux_privesc": [
        "sudo misconfiguration GTFOBins privilege escalation",
        "SUID SGID binary abuse privilege escalation",
        "cron job writable script root privilege escalation",
        "capabilities cap_setuid python privilege escalation",
        "writable /etc/passwd PATH hijack privilege escalation",
        "docker group lxd container escape root",
        "kernel exploit dirtycow pwnkit polkit CVE",
        "NFS no_root_squash privilege escalation",
    ],
    "web_exploitation": [
        "SQL injection union based authentication bypass",
        "local file inclusion remote file inclusion LFI RFI",
        "server side template injection SSTI RCE",
        "file upload bypass webshell remote code execution",
        "command injection OS command RCE web",
        "insecure deserialization RCE",
        "XXE external entity injection",
    ],
    "ad_attacks": [
        "Kerberoasting service ticket crack",
        "AS-REP roasting no preauth",
        "DCSync replication dump ntds",
        "BloodHound attack path ACL abuse",
        "ADCS certificate abuse ESC1 ESC8",
        "pass the hash overpass the hash",
        "unconstrained constrained delegation abuse",
    ],
}

# Intent detection: (intent_key, regex).  First match(es) fire; multiple may.
_INTENT_PATTERNS = [
    ("windows_privesc", r"\bwindows\b.*(priv\s*esc|privilege\s+escalation|privesc)"
                        r"|(priv\s*esc|privilege\s+escalation).*\bwindows\b"),
    ("linux_privesc", r"\blinux\b.*(priv\s*esc|privilege\s+escalation|privesc)"
                     r"|(priv\s*esc|privilege\s+escalation).*\blinux\b"),
    ("web_exploitation", r"web\s+app|web\s+application|\bfoothold\b|initial\s+access|"
                        r"\bweb\b.*(exploit|attack|vuln|cheat|technique|foothold)|website"),
    ("ad_attacks", r"\bactive\s+directory\b|\bAD\b.*(attack|abuse)|\bkerberos\b"),
]

_BROAD_MARKERS = re.compile(
    r"\bcheat\s*sheet\b|\bcheatsheet\b|\ball\b|\bevery\b|\blist\b|\bsummar|"
    r"\bwhat\s+(are|were)\b|\btechniques?\b|\bways?\s+to\b|\boverview\b",
    re.IGNORECASE,
)


def looks_broad(query: str) -> bool:
    """Heuristic: is this a broad, coverage-seeking question?"""
    return bool(_BROAD_MARKERS.search(query))


def heuristic_expand(query: str, max_subqueries: int = 8) -> List[str]:
    """Expand a broad query using the curated seed table."""
    q = query.lower()
    fired: List[str] = []
    for intent, pat in _INTENT_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            fired.extend(_SEEDS[intent])

    # Generic privilege-escalation query with no OS specified -> use both.
    if not fired and re.search(r"priv\s*esc|privilege\s+escalation", q):
        fired.extend(_SEEDS["windows_privesc"][:5])
        fired.extend(_SEEDS["linux_privesc"][:3])

    # De-dup while preserving order, always keep the original query first.
    out, seen = [query], {query.lower()}
    for s in fired:
        if s.lower() not in seen:
            out.append(s)
            seen.add(s.lower())
        if len(out) >= max_subqueries + 1:
            break
    return out


def llm_expand(query: str, cfg: Config, max_subqueries: int = 8) -> List[str]:
    """Ask the configured LLM for focused sub-queries (best-effort)."""
    from .synthesis import _get_llm_client  # lazy to avoid import cycle

    client = _get_llm_client(cfg)
    if client is None:
        return heuristic_expand(query, max_subqueries)

    prompt = (
        "You are helping a retrieval system answer an offensive-security "
        "question by breaking it into focused sub-queries.\n"
        f'Original question: "{query}"\n\n'
        f"Write up to {max_subqueries} short, specific search queries that "
        "together cover the distinct technique families needed to answer it. "
        "Use precise jargon (tool names, CVE ids, technique names). "
        "Return ONE query per line, no numbering, no commentary."
    )
    try:
        text = client.complete(
            system="You expand broad security questions into focused search sub-queries.",
            user=prompt,
            max_tokens=400,
            temperature=0.0,
        )
        subs = [ln.strip("-*0123456789. \t") for ln in text.splitlines() if ln.strip()]
        out, seen = [query], {query.lower()}
        for s in subs:
            if s and s.lower() not in seen:
                out.append(s)
                seen.add(s.lower())
        return out[: max_subqueries + 1]
    except Exception:
        return heuristic_expand(query, max_subqueries)


def expand_query(query: str, cfg: Config, use_llm: bool = False,
                 max_subqueries: int = 8) -> List[str]:
    """
    Return an expansion set for `query`.

    A narrow/specific query is returned unchanged (as a 1-element list) so we
    never dilute a precise jargon lookup.  Only broad queries are expanded.
    """
    if not looks_broad(query):
        return [query]
    if use_llm:
        return llm_expand(query, cfg, max_subqueries)
    return heuristic_expand(query, max_subqueries)
