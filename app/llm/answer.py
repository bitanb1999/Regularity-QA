"""Answer prompt and the post-generation citation check.

The model sees sources labelled [S1]..[Sn] (short labels: long chunk ids get mangled when
copied) and must cite every claim. `check` then decides deterministically whether the answer
is usable: no valid citation -> rejected; citations to labels that weren't provided -> stripped
and flagged; many uncited sentences -> flagged.
"""
import re
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["answered", "flagged", "refused", "rejected"]

NOT_FOUND = "NOT_FOUND:"
REJECTED_MSG = (
    "I couldn't produce an answer that is supported by citations to the indexed warning letters, "
    "so I'm not showing one. Try rephrasing or narrowing the question."
)
MAX_SOURCE_CHARS = 900
MAX_GRAPH_ROWS = 20
MIN_COVERAGE = 0.6

SYSTEM = f"""You answer questions about FDA warning letters for compliance professionals.

Rules:
- Use ONLY the numbered sources and graph results provided. No outside knowledge.
- Put a citation like [S2] or [S1, S4] at the end of every sentence or bullet that states a fact.
  Graph results list the source labels that support each row; cite those labels.
- If the sources do not contain the answer, reply with "{NOT_FOUND}" followed by one sentence on
  what is missing. Do not guess.
- If the sources answer only part of the question, answer that part and say what is not covered.
- "(b)(4)" and similar are redactions; never speculate about redacted content.
- Name companies and letter dates when relevant. Be concise; use bullets for lists.
- No summary or background sections: every line must come from the sources and carry a citation."""


@dataclass
class PromptSource:
    label: str
    company: str | None
    issue_date: str | None
    section: str
    text: str


def build_prompt(question: str, sources: list[PromptSource], graph_block: str | None) -> str:
    parts = [f"Question: {question}\n"]
    if graph_block:
        parts.append(graph_block)
    parts.append("Sources:")
    for s in sources:
        text = s.text if len(s.text) <= MAX_SOURCE_CHARS else s.text[:MAX_SOURCE_CHARS] + " …"
        parts.append(f"[{s.label}] {s.company} — warning letter {s.issue_date} — {s.section}\n{text}\n")
    return "\n".join(parts)


def graph_block(template: str, rows: list[dict], labels_by_chunk: dict[str, str]) -> str:
    lines = [f"Graph results (template: {template}, {len(rows)} rows):"]
    for row in rows[:MAX_GRAPH_ROWS]:
        cite = sorted({labels_by_chunk[c] for c in row.get("chunk_ids", []) if c in labels_by_chunk},
                      key=lambda x: int(x[1:]))
        cite_s = f" [{', '.join(cite)}]" if cite else " (evidence not loaded; do not cite this row)"
        if "title" in row:
            num = f"#{row['number']} " if row.get("number") is not None else ""
            regs = ", ".join(row.get("regulations") or []) or "none"
            lines.append(
                f"- {row['company']} (letter {row['issue_date']}, site {row['site']}): observation {num}"
                f"\"{row['title']}\". {row['summary'][:220]} Regulations: {regs}. "
                f"Topics: {', '.join(row.get('topics') or [])}.{cite_s}"
            )
        else:
            key = row.get("regulation") or row.get("topic")
            lines.append(f"- {key}: {row['observations']} observations; companies: "
                         f"{', '.join(row['companies'])}.{cite_s}")
    if len(rows) > MAX_GRAPH_ROWS:
        lines.append(f"- ({len(rows) - MAX_GRAPH_ROWS} more rows not shown)")
    return "\n".join(lines) + "\n"


# Models pad and space citations unpredictably ("[ S3 ]", "[S1,\u202fS4]", "【S2】"); \s also
# matches Unicode spaces. Matches are rewritten to the canonical "[S1, S4]" form.
_CITE = re.compile(r"[\[【]\s*((?:S\s*)?\d+(?:\s*[,;]\s*(?:S\s*)?\d+)*)\s*[\]】]")
_UNITS = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Check:
    status: Status
    answer: str
    cited: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    coverage: float = 0.0


def _labels(group: str) -> list[str]:
    return [f"S{n}" for n in re.findall(r"\d+", group)]


_INVISIBLE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")  # zero-width chars: not matched by \s


_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}")
_HEADING = re.compile(r"^\s*(#+\s.*|\*\*[^*]+\*\*:?)\s*$")


def _claim_text(text: str) -> str:
    """Text with markdown structure (headings, table header + rule rows) removed: not claims."""
    lines = text.split("\n")
    skip = {j for i, ln in enumerate(lines) if _TABLE_RULE.match(ln) for j in (i - 1, i)}
    return "\n".join(ln for i, ln in enumerate(lines) if i not in skip and not _HEADING.match(ln))


def check(text: str, available: set[str]) -> Check:
    text = _INVISIBLE.sub("", text).strip()
    if text.upper().startswith(NOT_FOUND):
        return Check("refused", text[len(NOT_FOUND):].strip() or "Not found in the indexed letters.",
                     flags=["model_not_found"])

    text = _CITE.sub(lambda m: f"[{', '.join(_labels(m.group(1)))}]", text)
    text = re.sub(r"[\u202f\u00a0]+(?=\[S)", " ", text)
    used = [lab for m in _CITE.finditer(text) for lab in _labels(m.group(1))]
    invalid = sorted(set(used) - available, key=lambda x: int(x[1:]))
    valid = sorted(set(used) & available, key=lambda x: int(x[1:]))
    flags: list[str] = []

    if invalid:
        flags.append(f"invalid_citations:{','.join(invalid)}")

        def keep_valid(m: re.Match) -> str:
            kept = [lab for lab in _labels(m.group(1)) if lab in available]
            return f"[{', '.join(kept)}]" if kept else ""
        text = _CITE.sub(keep_valid, text)

    if not valid:
        return Check("rejected", REJECTED_MSG, flags=[*flags, "no_valid_citations"])

    units = [u for u in _UNITS.split(_claim_text(text)) if len(u.split()) >= 6 and not u.rstrip().endswith(":")]
    coverage = sum(bool(_CITE.search(u)) for u in units) / len(units) if units else 1.0
    if coverage < MIN_COVERAGE:
        flags.append(f"uncited_claims:{coverage:.0%}")
    return Check("flagged" if flags else "answered", text, valid, flags, coverage)
