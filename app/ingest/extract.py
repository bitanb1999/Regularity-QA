"""LLM entity extraction for one warning letter, verified against the source text.

Three safeguards around the LLM call:
  1. Completeness: every numbered violation paragraph ("3. Your firm failed to ...") must start
     an observation; any the model skipped or merged are re-requested in a focused follow-up.
  2. Claims: a regulation is kept only if the regex scanner also finds it in the letter, and
     is_repeat only if the observation's text mentions a prior inspection/citation.
  3. Spans: clamped, de-overlapped and extended over trailing request lists.

Raw LLM output is cached in data/extracted/<doc_id>.json (bump PROMPT_VERSION to invalidate);
the checks above re-run on every load, so they can change without new LLM calls.
"""
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from app.config import settings
from app.ingest.citations import canonical, find_citations
from app.ingest.parse import NUMBERED, Block, ParsedDoc
from app.llm.client import structured
from app.models.extraction import Extraction, GapFill, ObservationX

log = logging.getLogger(__name__)

PROMPT_VERSION = 2

# "Repeat" in FDA usage means cited at an earlier inspection, not "happened several times".
REPEAT = re.compile(
    r"\brepeat (observation|violation|finding|deviation)s?\b|\b(previous|prior|last) (inspection|warning letter)"
    r"|\bpreviously (cited|observed|identified)\b|\b(also|similar\w*) cited\b", re.IGNORECASE)
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "extracted"

TOPICS = """Topics (pick 1-3 that describe the violation itself):
data_integrity: incomplete/altered/deleted records, shared logins, audit trails, unrecorded tests
laboratory_controls: release testing, test methods, lab records
oos_investigation: investigating out-of-specification results, discrepancies, CAPA
quality_unit: quality unit authority, review and oversight
component_testing: identity/purity testing of incoming ingredients
process_validation: validating manufacturing processes
cleaning_validation: validating equipment cleaning
stability_testing: stability programs, expiry dating
storage_conditions: temperature/humidity/light control in storage
contamination_control: preventing cross-contamination or adulteration in processing
pathogen_contamination: pathogens (Salmonella, Listeria, etc.) found in product or facility
sanitation_hygiene: cleaning/sanitizing, employee hygiene, insanitary conditions
equipment_facilities: equipment design or maintenance, building condition
hazard_analysis: hazard analysis, HACCP plans, preventive controls
supplier_verification: FSVP and other verification of foreign or ingredient suppliers
labeling_misbranding: label content, misbranding charges
unapproved_drug: marketing drugs without required FDA approval
false_advertising: false or misleading promotional or website claims
registration_listing: facility registration or drug listing
other: none of the above fits"""

SYSTEM = f"""You extract structured facts from FDA warning letters for a compliance database.
Rules:
- Use only what the letter states. Never infer facts that are not written.
- The letter is given as numbered blocks: [index] (kind) text. "(b)(4)" etc. are redactions.
- An observation is one distinct violation FDA says the firm committed. EVERY numbered item
  ("1. Your firm failed to ...") is its own observation; letters often restart numbering under
  a second heading (e.g. "Misbranding Violations"), and those count too. An unnumbered section
  that charges a violation (e.g. a misbranding or advertising claim) is one observation.
- Advisory or boilerplate sections are NOT observations: CGMP consultant recommendations,
  quality-systems advice, cosmetics notices, response instructions, conclusions.
- An observation's span runs from the block where it starts through its supporting evidence,
  FDA's assessment of the firm's response and any "In response to this letter, provide" list,
  ending before the next observation or the next non-observation section. Spans must not overlap.
- regulations: only citations written in the letter that apply to that observation, formatted
  '21 CFR 211.192(b)', '21 CFR Part 117' or 'FD&C Act 501(a)(2)(B)'. Use [] if none applies.
- is_repeat: true only if the letter says this violation was also cited at a previous
  inspection ("repeat observation"). Repetition within the same inspection is not a repeat.

{TOPICS}"""

GAP_SYSTEM = f"""You extract FDA warning-letter violations for a compliance database.
You are given part of a letter as numbered blocks and a list of block indices. Each listed block
begins one numbered violation. Return exactly one observation per listed block, with
first_block equal to that index, and a span that ends before the next violation or section.
Use only what the text states. Format regulations as '21 CFR 211.192(b)', '21 CFR Part 117'
or 'FD&C Act 403(k)'.

{TOPICS}"""


class LLMOutput(BaseModel):
    """What the LLM returned (cached). Everything derived from it is recomputed on load."""
    doc_id: str
    model: str
    prompt_version: int
    raw: Extraction
    gap: list[ObservationX] = []


class ExtractedDoc(BaseModel):
    doc_id: str
    extraction: Extraction
    dropped_citations: list[dict]
    gap_filled: list[int]


def _letter_end(doc: ParsedDoc) -> int:
    return next(
        (b.idx for b in doc.blocks if b.kind == "heading" and b.text.lower() == "conclusion"),
        len(doc.blocks),
    )


def _render(doc: ParsedDoc, blocks: list[Block] | None = None, char_budget: int = 16000) -> str:
    """Numbered blocks, shrunk to fit the provider's per-request token limit.

    Blocks after the "Conclusion" heading are boilerplate and are omitted; if still too long,
    each block is truncated to a decreasing cap. Indices are preserved either way, so spans
    the LLM returns still refer to the full text.
    """
    m = doc.meta
    head = f"Letter: {m['title']}\nIssued: {m['issue_date']}\nSubject: {m['subject']}\n\n"
    if blocks is None:
        end = _letter_end(doc)
        blocks = [b for b in doc.blocks if b.idx < end or b.kind == "footnote"]

    def clip(b: Block, cap: int | None) -> str:
        if cap is not None and b.kind in ("item", "footnote"):
            cap = min(cap, 140)  # response-request lists carry little extractable fact
        return b.text if cap is None or len(b.text) <= cap else b.text[:cap] + " …"

    for cap in (None, 900, 600, 400, 250):
        text = head + "\n".join(f"[{b.idx}] ({b.kind}) {clip(b, cap)}" for b in blocks)
        if len(text) <= char_budget:
            break
    return text


def _numbered_starts(doc: ParsedDoc) -> list[int]:
    end = _letter_end(doc)
    return [b.idx for b in doc.blocks if b.idx < end and b.kind == "para" and NUMBERED.match(b.text)]


def _missing(doc: ParsedDoc, obs: list[ObservationX]) -> list[int]:
    starts = {o.first_block for o in obs}
    return [i for i in _numbered_starts(doc) if i not in starts]


def _request_gaps(doc: ParsedDoc, missing: list[int]) -> list[ObservationX]:
    log.warning("%s: numbered violations not extracted separately: %s", doc.meta["id"], missing)
    # Context: from the heading above the first gap to the end of the letter body.
    lo = max((b.idx for b in doc.blocks if b.kind == "heading" and b.idx < missing[0]), default=0)
    region = doc.blocks[lo : _letter_end(doc)]
    prompt = _render(doc, region) + f"\n\nBlocks that each begin a violation: {missing}"
    return [o for o in structured(GAP_SYSTEM, prompt, GapFill).observations if o.first_block in missing]


def _merge_gaps(doc: ParsedDoc, obs: list[ObservationX], new: list[ObservationX]) -> list[ObservationX]:
    """An existing observation that swallowed new starts and isn't itself a numbered violation was
    a container (e.g. a section heading); replace it. Numbered ones get trimmed by _fix_spans."""
    numbered = set(_numbered_starts(doc))
    new_starts = {o.first_block for o in new}
    kept = [o for o in obs
            if o.first_block in numbered or not any(o.first_block < s <= o.last_block for s in new_starts)]
    merged = kept + new
    if still := _missing(doc, merged):
        log.error("%s: still missing numbered violations %s", doc.meta["id"], still)
    return merged


def _fix_spans(obs: list[ObservationX], doc: ParsedDoc) -> list[ObservationX]:
    """Clamp spans to the document, remove overlaps, and absorb trailing list items
    (the tail of a "please provide" list) that the LLM tends to leave out by one."""
    blocks = doc.blocks
    numbered = set(_numbered_starts(doc))
    ordered = sorted(obs, key=lambda o: o.first_block)
    out: list[ObservationX] = []
    for i, o in enumerate(ordered):
        first = max(o.first_block, out[-1].last_block + 1 if out else 0)
        stop = ordered[i + 1].first_block if i + 1 < len(ordered) else len(blocks)
        last = min(o.last_block, stop - 1)
        between = blocks[last + 1 : stop]
        if stop in numbered and not any(b.kind == "heading" for b in between):
            last = stop - 1  # text between two numbered violations belongs to the earlier one
        while last + 1 < stop and blocks[last + 1].kind == "item":
            last += 1
        if first > last:
            log.warning("dropping observation with empty span: %s", o.title)
            continue
        out.append(o.model_copy(update={"first_block": first, "last_block": last}))
    return out


def _verify(doc: ParsedDoc, obs: list[ObservationX]) -> tuple[list[ObservationX], list[dict]]:
    """Keep only citations that exist in the letter; add any written inside an observation's span."""
    doc_cites = find_citations(doc.body)
    dropped: list[dict] = []
    out = []
    for o in _fix_spans(obs, doc):
        span_text = "\n".join(b.text for b in doc.blocks[o.first_block : o.last_block + 1])
        kept: list[str] = []
        for raw in o.regulations:
            cid = canonical(raw)
            if cid and cid in doc_cites:
                kept.append(cid)
            else:
                dropped.append({"observation": o.title, "citation": raw, "canonical": cid})
        regs = sorted(set(kept) | find_citations(span_text))
        number = o.number
        if m := NUMBERED.match(doc.blocks[o.first_block].text):
            number = int(m[1])
        is_repeat = o.is_repeat and bool(REPEAT.search(span_text))
        if o.is_repeat and not is_repeat:
            dropped.append({"observation": o.title, "claim": "is_repeat"})
        out.append(o.model_copy(update={
            "number": number, "regulations": regs, "topics": list(dict.fromkeys(o.topics)),
            "is_repeat": is_repeat}))
    return out, dropped


def _llm(doc: ParsedDoc, refresh: bool) -> LLMOutput:
    cache = CACHE_DIR / f"{doc.meta['id']}.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        if data.get("prompt_version") == PROMPT_VERSION and data.get("model") == settings.extraction_model:
            data.setdefault("raw", data.get("extraction"))  # pre-LLMOutput cache files
            return LLMOutput.model_validate(data)

    raw = structured(SYSTEM, _render(doc), Extraction)
    missing = _missing(doc, raw.observations)
    out = LLMOutput(
        doc_id=doc.meta["id"], model=settings.extraction_model, prompt_version=PROMPT_VERSION,
        raw=raw, gap=_request_gaps(doc, missing) if missing else [],
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out.model_dump(mode="json"), indent=2))
    return out


def extract(doc: ParsedDoc, refresh: bool = False) -> ExtractedDoc:
    llm = _llm(doc, refresh)
    obs = _merge_gaps(doc, llm.raw.observations, llm.gap) if llm.gap else llm.raw.observations
    obs, dropped = _verify(doc, obs)
    for d in dropped:
        log.info("%s: dropped unverified %s", doc.meta["id"], d.get("citation") or d.get("claim"))
    return ExtractedDoc(
        doc_id=doc.meta["id"],
        extraction=llm.raw.model_copy(update={"observations": obs}),
        dropped_citations=dropped,
        gap_filled=sorted(o.first_block for o in llm.gap),
    )
