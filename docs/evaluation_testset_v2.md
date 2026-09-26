# HTB RAG — Evaluation Test Set & Hand-Curated Answer Key (v2)

**Author role:** Offensive Security Specialist / Technical Evaluator
**Corpus:** `0xh7ml/htb-wiki` `raw/` — 513 HTB write-ups (0xdf style)
**Purpose:** Score a RAG system on retrieval *and* synthesis. Every "Ground Truth Machines"
list and command below was derived by grepping/reading the raw `.md` files, **not** by
trusting the RAG system.

**Composition:** 5 Broad · 7 Narrow · 3 Edge (alias, not-present, ambiguous scope).

**How to read the answer keys.** Machine lists are *representative and verified* — where a
technique appears on 40+ boxes the list is a curated sample of clear demonstrators plus a
count, so precision (are retrieved machines on-topic?) is the primary metric and recall is
read as coverage. Commands are quoted verbatim from the write-ups where a question asks
"what commands were used".

---

## Category A — Broad / Cheatsheet (5)

### B1
| Field | Value |
|---|---|
| **ID / Text** | **B1** — "Summarize the Windows privilege-escalation techniques found in the write-ups." |
| **Category** | Broad / Cheatsheet |
| **Gold Answer Key** | Answer must be **grouped by technique family**, each with a 1–3 sentence explanation and machine citations: **(1) SAM/SYSTEM hive dump** → `reg save` the hives + `secretsdump.py -sam -system LOCAL`; **(2) Potato / `SeImpersonatePrivilege` abuse** → JuicyPotato/RoguePotato/PrintSpoofer to SYSTEM; **(3) ADCS certificate abuse (ESC1)** → enrol cert for privileged principal, pass-the-cert; **(4) Kernel / missing-patch exploits** → Watson / exploit-suggester → MS-xx; **(5) DCSync / ACL abuse** → replicate directory changes to dump hashes. A strong answer names ≥4 families and cites a distinct machine per family. |
| **Ground Truth Machines** | SAM hive: **Bastion, Cicada, Omni**; Potato: **Json, Worker, Conceal**; ADCS: **Authority, Escape, Sendai**; Kernel: **Grandpa, Arctic, Bounty**; DCSync/ACL: **Forest, Blackfield** |
| **Evaluation Target** | Retrieval **diversity** across technique families (MMR / per-machine cap working); synthesis **grouping + per-family citation**. Fails if answer is dominated by one family or one machine. |

### B2
| Field | Value |
|---|---|
| **ID / Text** | **B2** — "Give me a Linux privilege-escalation cheatsheet based on these boxes." |
| **Category** | Broad / Cheatsheet |
| **Gold Answer Key** | Families: **(1) `sudo` misconfig / GTFOBins** → `sudo -l` then abuse an allowed binary; **(2) SUID/SGID binaries** → `find / -perm -4000` then GTFOBins; **(3) writable cron / scripts**; **(4) Linux capabilities** (`getcap`, e.g. `cap_setuid` on python); **(5) PATH / wildcard injection**. Each family explained + cited. |
| **Ground Truth Machines** | sudo/GTFOBins: **Admirer, Shocker (perl), Bashed, Nibbles**; SUID: **Valentine, Irked, Sunday, Titanic**; cron: **Bashed**; capabilities: **Bank/others** |
| **Evaluation Target** | Same as B1 but on the Linux half; also tests the OS-classifier isn't leaking Windows content into a Linux query. |

### B3
| Field | Value |
|---|---|
| **ID / Text** | **B3** — "What Active Directory attack techniques appear across the corpus?" |
| **Category** | Broad / Cheatsheet |
| **Gold Answer Key** | **(1) Kerberoasting** (`GetUserSPNs.py -request`); **(2) AS-REP roasting** (`GetNPUsers.py`); **(3) DCSync** (`secretsdump -just-dc`); **(4) ACL abuse via BloodHound** (GenericWrite/GenericAll/WriteDACL → targeted kerberoast / shadow creds); **(5) ADCS ESC1**. Cite ≥1 machine per technique. |
| **Ground Truth Machines** | **Forest** (AS-REP + DCSync), **Sauna** (AS-REP+Kerberoast), **Blackfield** (AS-REP, LSASS, backup→ntds), **Active** (Kerberoast + GPP), **Escape** (Kerberoast + ADCS), **Administrator** (DCSync/ACL) |
| **Evaluation Target** | Coverage across the AD kill-chain; correct machine→technique mapping (does it put Forest under AS-REP, not under LFI?). |

### B4
| Field | Value |
|---|---|
| **ID / Text** | **B4** — "Summarize the web-application initial-foothold techniques used to get the first shell." |
| **Category** | Broad / Cheatsheet |
| **Gold Answer Key** | **(1) LFI / log poisoning** → include a poisoned log/UA to run PHP; **(2) SQL injection** → auth bypass / dump / `--os-shell`; **(3) SSTI** → `{{7*7}}` → RCE; **(4) File-upload → webshell**; **(5) Command injection**. Each cited. |
| **Ground Truth Machines** | LFI: **Poison, Bart, Nineveh, Beep**; SQLi: **Fighter, Falafel**; SSTI: **(template-injection boxes)**; upload/webshell: **(upload boxes)**; cmd-injection: **(injection boxes)** |
| **Evaluation Target** | Breadth across web vuln classes; synthesis must not conflate a foothold technique with a privesc one. |

### B5
| Field | Value |
|---|---|
| **ID / Text** | **B5** — "Summarize the credential-dumping / hash-extraction techniques demonstrated." |
| **Category** | Broad / Cheatsheet |
| **Gold Answer Key** | **(1) SAM/SYSTEM hive** → `secretsdump -sam SAM -system SYSTEM LOCAL`; **(2) LSASS memory** → `procdump`/`comsvcs.dll` dump → `pypykatz`/Mimikatz; **(3) NTDS.dit via backup privs** → `diskshadow`/`SeBackupPrivilege` → `secretsdump -ntds`; **(4) DCSync** → `secretsdump -just-dc`; **(5) GPP cpassword** → decrypt `Groups.xml` cpassword. Cite per technique. |
| **Ground Truth Machines** | SAM: **Bastion**; LSASS: **Blackfield** (`lsass.zip` → pypykatz); NTDS/backup: **Blackfield, APT** (`ntds.dit`); DCSync: **Forest, Sauna**; GPP cpassword: **Active, Giveback** |
| **Evaluation Target** | Fine-grained distinction between *five different* credential-access methods that all yield hashes; tests that synthesis separates them rather than merging into "dump hashes". |

---

## Category B — Narrow / Specific (7)

### N1
| Field | Value |
|---|---|
| **ID / Text** | **N1** — "Which machine demonstrated SAM hive dumping and what commands were used?" |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | **Bastion**: mount the VHD backups from an open SMB share, copy the `SAM`, `SECURITY`, `SYSTEM` hives out of `Windows/System32/config`, then run offline: `secretsdump.py -sam SAM -security SECURITY -system SYSTEM LOCAL` to recover local NT hashes. Also acceptable: registry `reg save hklm\sam` on other boxes. |
| **Ground Truth Machines** | **Bastion** (primary), **Cicada, Omni, Acute, Baby, Freelancer, Mist, Outdated** |
| **Evaluation Target** | Precise machine+command retrieval; synthesis must reproduce the actual `secretsdump` invocation, not a generic description, and cite `(seen on: Bastion)`. |

### N2
| Field | Value |
|---|---|
| **ID / Text** | **N2** — "How does JuicyPotato work and which machines use it? Include the exploitation steps." |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | Requires `SeImpersonatePrivilege` (default for many service accounts, e.g. IIS/MSSQL). Steps: drop `JuicyPotato.exe` + a payload (`nc64.exe` + `rev.bat`), pick a **CLSID** mapped to SYSTEM for that Windows version, run `JuicyPotato -l <port> -p <payload> -t * -c {CLSID}` to abuse the BITS/DCOM→NTLM reflection and get a SYSTEM shell. Note the Server 2019 limitation (use RoguePotato/PrintSpoofer instead). |
| **Ground Truth Machines** | **Json, Worker, Conceal, Tally, Fighter, Bruno, Cereal, Hackback, Perspective, Shibuya** (13 total) |
| **Evaluation Target** | Mechanism-level synthesis (CLSID selection, SeImpersonate precondition) grounded only in context; multi-machine citation. |

### N3
| Field | Value |
|---|---|
| **ID / Text** | **N3** — "Which machines involve Kerberoasting and what tool/command is used?" |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | Impacket `GetUserSPNs.py -request -dc-ip <DC> <domain>/<user>` to request TGS tickets for SPN-bound accounts, then crack offline with hashcat mode `13100`. On **Active**: `GetUserSPNs.py -request -dc-ip 10.10.10.100 active.htb/SVC_TGS -save -outputfile GetUserSPNs.out` → crack to recover the `administrator` SPN account. |
| **Ground Truth Machines** | **Active, Blackfield, Sizzle, Search, Rebound, Pivotapi** (+ Forest/Sauna via Kerberoast) |
| **Evaluation Target** | Tool+command fidelity; does not confuse Kerberoasting with AS-REP roasting (a common retrieval false-positive). |

### N4
| Field | Value |
|---|---|
| **ID / Text** | **N4** — "Which machines use AS-REP roasting and how?" |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | Target accounts with "Do not require Kerberos pre-auth" set. `GetNPUsers.py <domain>/ -usersfile users.txt -no-pass -dc-ip <DC>` (or with creds) to obtain AS-REP hashes, crack with hashcat mode `18200`. **Forest**: enumerate users over RPC → `GetNPUsers` → crack `svc-alfresco`. |
| **Ground Truth Machines** | **Forest, Sauna, Absolute, Jab, Intelligence, Blackfield, Multimaster, Mantis** |
| **Evaluation Target** | Disambiguation from Kerberoasting; correct `GetNPUsers` vs `GetUserSPNs` mapping and hashcat mode. |

### N5
| Field | Value |
|---|---|
| **ID / Text** | **N5** — "Which machine demonstrates ADCS ESC1 and what are the exact steps?" |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | **Authority**: find AD CS template where *any domain computer* can enrol and the template allows `enrollee-supplies-subject` (ESC1). Add a fake machine account, request a cert specifying an alternate UPN (the DC/administrator) with `certipy req ... -template <T> -upn administrator`, then **pass-the-cert** (`certipy auth`) to obtain a TGT / dump hashes. Cite Certipy commands. |
| **Ground Truth Machines** | **Authority** (primary), **Escape, Sendai, Manager, Certified, Fluffy, Mist, Shibuya** |
| **Evaluation Target** | Deep single-technique retrieval + faithful multi-step synthesis (add computer → request → auth) without inventing ESC-numbers not in context. |

### N6
| Field | Value |
|---|---|
| **ID / Text** | **N6** — "Which machines perform a DCSync attack and what command dumps the hashes?" |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | With replication rights (an account holding `DS-Replication-Get-Changes`/`-All`, often gained via ACL abuse), run `secretsdump.py <domain>/<user>:<pass>@<DC> -just-dc` (or `-just-dc-ntlm`) to replicate and dump all domain hashes incl. `krbtgt`. **Forest**: grant DCSync via ACL then dump the administrator hash. |
| **Ground Truth Machines** | **Forest, Sauna, Authority, Administrator, Rebound, Mist, Multimaster, Scepter, Flight** |
| **Evaluation Target** | Precondition reasoning (needs replication ACL) + exact `-just-dc` command; correct machine mapping. |

### N7
| Field | Value |
|---|---|
| **ID / Text** | **N7** — "Which machines abuse `SeImpersonatePrivilege`, and with which tools (PrintSpoofer / potato family)?" |
| **Category** | Narrow / Specific |
| **Gold Answer Key** | Service accounts (IIS `iis apppool`, `mssql-svc`) hold `SeImpersonatePrivilege`. Tools by OS build: **JuicyPotato** (≤ Server 2016), **RoguePotato / PrintSpoofer / GodPotato** (Server 2019+). Flow: confirm with `whoami /priv`, run the tool with a reverse-shell payload to get SYSTEM. |
| **Ground Truth Machines** | **APT, Bounty, Querier, Silo, Escape, Rainbow, Mailing, Arkham, Worker, Json** (~40 across corpus) |
| **Evaluation Target** | Retrieval of a privilege→tool family relationship; synthesis must map the right tool to the right Windows version and cite multiple boxes. |

---

## Category C — Edge Cases (3)

### E1 — Synonym / alias variation
| Field | Value |
|---|---|
| **ID / Text** | **E1** — "Show me **PE** techniques on Windows boxes that abuse **impersonation tokens** — the '**potato**' attacks (rotten/juicy/rogue/god potato)." |
| **Category** | Edge / Alias |
| **Gold Answer Key** | Interpret **PE = privilege escalation**, "impersonation tokens" = `SeImpersonatePrivilege`, and the "potato" aliases (RottenPotato, JuicyPotato, RoguePotato, GodPotato, SweetPotato, PrintSpoofer as a cousin). Same substance as **N7**: enumerate `whoami /priv`, pick the potato variant for the OS build, get SYSTEM. Must **not** return Linux privesc or the vegetable/food false-positives. |
| **Ground Truth Machines** | Same as N7 set — **Json, Worker, Conceal, APT, Bounty, Querier, Silo, Cereal** |
| **Evaluation Target** | **Vocabulary robustness**: does the embedder/expander map the abbreviation "PE" and the alias cluster to the canonical `SeImpersonatePrivilege`/potato content? Tests query-expansion + semantic matching, not just literal BM25. |

### E2 — Technique NOT present (hallucination / zero-result test)
| Field | Value |
|---|---|
| **ID / Text** | **E2** — "Which machines exploit the **XZ Utils backdoor (CVE-2024-3094)** and how is the malicious `liblzma` triggered?" |
| **Category** | Edge / Not-Present |
| **Gold Answer Key** | **Correct answer = "Information not found in context."** The XZ Utils / `liblzma` supply-chain backdoor (CVE-2024-3094) does **not** appear in any write-up (grep over `raw/` = **0 files**). The system must **refuse to answer / return no grounded result** and must **not** fabricate a machine name, CVE detail, or `sshd`/`liblzma` mechanism from parametric memory. |
| **Ground Truth Machines** | **None** (verified: 0 corpus hits for `xz utils` / `cve-2024-3094`) |
| **Evaluation Target** | **Hallucination resistance / zero-result handling.** Pass = explicit "not found" + zero fabricated citations. Fail = any invented machine or confident CVE writeup. (Also a useful low-similarity-threshold check on the retriever.) |

### E3 — Ambiguous scoping (mixed Windows/Linux)
| Field | Value |
|---|---|
| **ID / Text** | **E3** — "How do I get **root** using **`sudo`** and **`SeImpersonatePrivilege`**?" |
| **Category** | Edge / Ambiguous scope |
| **Gold Answer Key** | The query mixes a **Linux** vector (`sudo -l` → GTFOBins → `root`) with a **Windows** vector (`SeImpersonatePrivilege` → potato/PrintSpoofer → SYSTEM; "root" is loosely SYSTEM). A good answer **separates the two OS contexts explicitly** — a Linux paragraph (sudo/GTFOBins boxes) and a Windows paragraph (SeImpersonate boxes) — rather than blending them into one incoherent chain. Each side cited to its own machines. |
| **Ground Truth Machines** | Linux/sudo: **Admirer, Shocker, Bashed, Nibbles**; Windows/SeImpersonate: **APT, Bounty, Querier, Worker** |
| **Evaluation Target** | **Scope disambiguation.** Tests whether retrieval returns *both* clusters (not just the dominant one) and whether synthesis recognises the two are different OSes and does not conflate `sudo` with `SeImpersonate` in a single fake exploit path. |

---

## Suggested scoring rubric

| Dimension | How to score |
|---|---|
| **Retrieval precision@k** | fraction of retrieved machines that are in the Ground-Truth set (primary metric for Narrow + E1/E3). |
| **Retrieval hit / coverage** | Broad: # distinct technique families represented in top-k; Narrow: is the primary machine present. |
| **Citation correctness** | every technique carries `(seen on: <Machine>)`; machine actually demonstrates it. |
| **Grounding / no-invention** | commands & CVEs appear in retrieved context; **E2 must yield "Information not found in context."** |
| **Scope handling (E3)** | both OS clusters returned and kept separate. |
| **Alias handling (E1)** | "PE"/"potato" resolved to SeImpersonate content. |
