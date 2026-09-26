#!/usr/bin/env python3
"""
Hand-derived answer-key builder.

Derives the gold machine set for each test question by grepping the RAW
write-ups (never by trusting the RAG system).  Each technique has an explicit,
auditable regex; a machine is 'gold' if its raw .md matches.  Broad cheatsheet
questions take the union of their constituent technique sets.

Run:  python scripts/build_answer_key.py --corpus htb-wiki/raw --out data/testset.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

# Per-technique detection patterns (case-insensitive, matched against full file).
TECH = {
    "juicypotato":   r"juicypotato",
    "sam_hive":      r"reg\s+save\s+hk|-sam\s+sam|secretsdump.*-sam|\bsam\b.*\bsystem\b.*hive|save.*\\sam",
    "adcs_esc1":     r"esc1|certipy|vulnerable\s+certificate\s+template|misconfigured\s+certificate",
    "kerberoast":    r"kerberoast",
    "asrep":         r"as-?rep\s*roast|asreproast|\bAS-REP\b",
    "acl_abuse":     r"genericwrite|genericall|writedacl|forcechangepassword|writeowner|addself",
    "seimpersonate": r"printspoofer|seimpersonate|juicypotato|roguepotato|godpotato|sweetpotato",
    "sudo":          r"gtfobins|sudo\s+-l|\(ALL\s*:?\s*ALL\)|NOPASSWD",
    "suid":          r"\bsuid\b|find\s+/\s+-perm.*[24]000|-perm\s+-[24]000",
    "log4shell":     r"log4j|log4shell|jndi:(ldap|rmi)",
    "dcsync":        r"dcsync|drsuapi|\-just-dc|replicating\s+directory\s+changes",
    "lfi":           r"local\s+file\s+inclusion|\bLFI\b|\.\./\.\./\.\./etc/passwd|log\s+poisoning",
    "sqli":          r"sql\s*injection|\bsqli\b|union\s+select|' or '?1'?='?1|sqlmap",
    "ssti":          r"server[- ]side\s+template\s+injection|\bssti\b|\{\{\s*7\s*\*\s*7\s*\}\}",
    "file_upload":   r"file\s+upload|upload.*(web\s*shell|\.php|\.aspx)|webshell",
    "cmd_injection": r"command\s+injection|os\s+command\s+injection|;\s*(id|whoami|nc)\b",
    "kernel_exploit":r"exploit\s+suggester|watson\b|\bMS1[0-9]-0|kernel\s+exploit|dirtycow|pwnkit",
}

# Which technique(s) each question's gold set is built from.
QUESTION_TECH = {
    "Q01": ["juicypotato", "seimpersonate", "sam_hive", "adcs_esc1", "kernel_exploit", "dcsync"],
    "Q02": ["sudo", "suid"],
    "Q03": ["kerberoast", "asrep", "dcsync", "acl_abuse", "adcs_esc1"],
    "Q04": ["lfi", "sqli", "ssti", "file_upload", "cmd_injection"],
    "Q05": ["juicypotato"],
    "Q06": ["sam_hive"],
    "Q07": ["adcs_esc1"],
    "Q08": ["kerberoast"],
    "Q09": ["asrep"],
    "Q10": ["acl_abuse"],
    "Q11": ["seimpersonate"],
    "Q12": ["sudo"],
    "Q13": ["suid"],
    "Q14": ["log4shell"],
    "Q15": ["dcsync"],
    "Q16": ["lfi"],
}


def slug_to_name(slug: str) -> str:
    return slug[0].upper() + slug[1:] if slug else slug


def derive(corpus: Path):
    files = sorted(corpus.rglob("*.md"))
    # machine slug -> set of techniques it matches
    machine_tech: dict = {}
    compiled = {k: re.compile(v, re.IGNORECASE) for k, v in TECH.items()}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        slug = re.sub(r"^htb-", "", f.stem)
        for tech, rx in compiled.items():
            if rx.search(text):
                machine_tech.setdefault(tech, set()).add(slug)
    return machine_tech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="htb-wiki/raw")
    ap.add_argument("--questions", default="data/testset.json")
    ap.add_argument("--out", default="data/testset.json")
    args = ap.parse_args()

    machine_tech = derive(Path(args.corpus))
    questions = json.loads(Path(args.questions).read_text())

    for item in questions:
        techs = QUESTION_TECH.get(item["id"])
        if not techs:
            continue
        gold = set()
        for t in techs:
            gold |= machine_tech.get(t, set())
        names = sorted(slug_to_name(s) for s in gold)
        item["gold_machines"] = names
        item["gold_count"] = len(names)
        item["derived_from"] = techs

    Path(args.out).write_text(json.dumps(questions, indent=2) + "\n")
    print(f"[+] Wrote {args.out}")
    for item in questions:
        print(f"  {item['id']}: {item.get('gold_count', 0):3d} gold machines "
              f"({'+'.join(item.get('derived_from', []))})")


if __name__ == "__main__":
    main()
