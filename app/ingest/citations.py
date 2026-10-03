"""Deterministic regulation-citation extraction and normalization.

Canonical ids (also the Neo4j Regulation keys):
  21 CFR section -> "21 CFR 211.192"     (paragraph refs like (b)(1) dropped)
  21 CFR part    -> "21 CFR Part 211"
  FD&C Act       -> "FD&C Act 501(a)"   (first subsection kept: 501(a) and 501(b) differ in meaning)

The LLM's citations are only kept if this module also finds them in the source text,
so the graph can't contain a regulation the letter never mentions.
"""
import re

_PARENS = re.compile(r"\([a-z0-9]{1,4}\)", re.IGNORECASE)
_SEP = r"(?:\s*(?:,|&|\band\b|\bor\b|\bthrough\b)\s*)"

# "21 CFR 211.22(a) & 211.22(d)", "21 CFR, parts 210 and 211", "21 CFR part 1, subpart L"
_CFR = re.compile(
    r"21\s*C\.?F\.?R\.?,?\s*(?P<body>(?:(?:§+\s*|parts?\s+)?\d+(?:\.\d+)?(?:\([a-z0-9]{1,4}\))*"
    r"(?:,?\s*subpart\s+[A-Z]\b)?" + _SEP + r"?)+)",
    re.IGNORECASE,
)
_FDC_SEC = r"\d{3}[A-Z]{0,2}(?:\([a-z0-9]{1,4}\))*"
_FDC_LIST = rf"(?P<list>{_FDC_SEC}(?:{_SEP}+{_FDC_SEC})*)"
_ACT = r"(?:FD&C|Federal Food,? Drug,? and Cosmetic)\s+Act|(?<=the )Act\b"
# "sections 502(a), 502(bb), and 201(n) of the FD&C Act" / "FD&C Act sections 512(a) and 502(f)"
_FDC_FWD = re.compile(
    rf"\bsections?\s+{_FDC_LIST}(?:\s*\(if applicable\))?\s+of\s+the\s+(?:{_ACT})", re.IGNORECASE
)
_FDC_REV = re.compile(
    rf"(?:FD&C\s+Act|Federal Food,? Drug,? and Cosmetic\s+Act),?\s+sections?\s+{_FDC_LIST}", re.IGNORECASE
)


def _cfr_ids(body: str) -> set[str]:
    out = set()
    for tok in re.findall(r"\d+(?:\.\d+)?", _PARENS.sub("", body)):
        out.add(f"21 CFR {tok}" if "." in tok else f"21 CFR Part {tok}")
    return out


def _fdc_ids(lst: str) -> set[str]:
    out = set()
    for sec, sub in re.findall(r"(\d{3}[A-Z]{0,2})((?:\([a-z0-9]{1,4}\))*)", lst, re.IGNORECASE):
        first = _PARENS.findall(sub)[:1]
        out.add(f"FD&C Act {sec.upper()}{first[0].lower() if first else ''}")
    return out


def find_citations(text: str) -> set[str]:
    text = text.replace("§", "§")
    ids: set[str] = set()
    for m in _CFR.finditer(text):
        ids |= _cfr_ids(m["body"])
    for rx in (_FDC_FWD, _FDC_REV):
        for m in rx.finditer(text):
            ids |= _fdc_ids(m["list"])
    return ids


def canonical(raw: str) -> str | None:
    """Normalize one citation string as the LLM wrote it ("21 CFR 211.192(b)", "FD&C Act 501(a)(2)(B)")."""
    raw = raw.strip()
    if re.search(r"C\.?F\.?R", raw, re.IGNORECASE):
        ids = sorted(find_citations(raw))
    elif re.search(r"FD&C|Cosmetic Act|^\s*(section\s+)?\d{3}", raw, re.IGNORECASE):
        m = re.search(_FDC_SEC, raw, re.IGNORECASE)
        ids = sorted(_fdc_ids(m.group(0))) if m else []
    else:
        ids = []
    return ids[0] if len(ids) == 1 else None


def parent(reg_id: str) -> str | None:
    """'21 CFR 211.192' -> '21 CFR Part 211'; 'FD&C Act 501(a)' -> 'FD&C Act 501'."""
    if m := re.fullmatch(r"21 CFR (\d+)\.\d+", reg_id):
        return f"21 CFR Part {m[1]}"
    if m := re.fullmatch(r"FD&C Act (\d{3}[A-Z]{0,2})\(.+\)", reg_id):
        return f"FD&C Act {m[1]}"
    return None
