from pathlib import Path

import pytest

from app.ingest.chunk import chunk
from app.ingest.extract import _fix_spans, _numbered_starts, _verify
from app.ingest.parse import parse
from app.models.extraction import ObservationX

LETTER = """<html><head>
<title>Acme Pharma Inc. - 700001 - 09/01/2026 | FDA</title>
<meta name="dcterms.title" content="Acme Pharma Inc. - 700001 - 09/01/2026">
<meta name="dcterms.description" content="CGMP/Finished Pharmaceuticals/Adulterated">
<meta property="og:url" content="https://www.fda.gov/warning-letters/acme-pharma-inc-700001-09012026">
<meta property="og:locality" content="Newark"><meta property="og:region" content="NJ">
</head><body><article>
<div class="col-md-8"><p>WARNING LETTER</p></div>
<div class="col-md-8">
  <dl><dt>Product:</dt><dd>Drugs</dd></dl>
  <p>September 1, 2026</p>
  <p>Dear Mr. Smith:</p>
  <p>FDA inspected your facility from March 2 to March 6, 2026.</p>
  <p><strong>Violations of the Federal Food, Drug, and Cosmetic Act</strong></p>
  <p><strong>1. Your firm failed to investigate OOS results (21 CFR 211.192).</strong></p>
  <p>Batch <span>(b)(4)</span> failed assay<sup>1</sup>.</p>
  <ul><li>Provide a CAPA plan.</li><li>Provide a retrospective review.</li></ul>
  <p><strong>2. Your firm failed to prevent unauthorized changes to records (21 CFR 211.68(b)).</strong></p>
  <p>Analysts shared passwords.</p>
  <p><strong>Conclusion</strong></p>
  <p>Respond within 15 working days.</p>
  <p>Sincerely,</p><p>/S/</p><p>Jane Doe, Director</p>
  <hr><p>1 See section 501(a)(2)(B) of the FD&amp;C Act.</p>
</div></article></body></html>"""


@pytest.fixture
def doc(tmp_path: Path):
    p = tmp_path / "acme-pharma-inc-700001-09012026.html"
    p.write_text(LETTER)
    return parse(p)


def _obs(first, last, regs=()):
    return ObservationX(number=None, section=None, title="t", summary="s", first_block=first,
                        last_block=last, regulations=list(regs), topics=["other"], is_repeat=False)


def test_parse_metadata_and_blocks(doc):
    assert doc.meta["company"] == "Acme Pharma Inc."
    assert str(doc.meta["issue_date"]) == "2026-09-01"
    kinds = [(b.kind, b.text[:20]) for b in doc.blocks]
    assert ("heading", "Conclusion") in kinds
    # inline redaction stays in its sentence; footnote marker is removed
    assert any(b.text == "Batch (b)(4) failed assay." for b in doc.blocks)
    # signature dropped, footnote kept
    assert not any("Jane Doe" in b.text for b in doc.blocks)
    assert doc.blocks[-1].kind == "footnote"


def test_numbered_starts(doc):
    texts = [doc.blocks[i].text[:2] for i in _numbered_starts(doc)]
    assert texts == ["1.", "2."]


def test_fix_spans_absorbs_trailing_items(doc):
    s1, s2 = _numbered_starts(doc)
    fixed = _fix_spans([_obs(s1, s1 + 1), _obs(s2, s2 + 1)], doc)
    assert fixed[0].last_block == s2 - 1  # the two "Provide ..." items were absorbed


def test_fix_spans_trims_overlap(doc):
    s1, s2 = _numbered_starts(doc)
    fixed = _fix_spans([_obs(s2, s2 + 1), _obs(s1, s2 + 1)], doc)
    assert [(o.first_block, o.last_block) for o in fixed] == [(s1, s2 - 1), (s2, s2 + 1)]


def test_verify_drops_unsupported_and_adds_span_citations(doc):
    s1, s2 = _numbered_starts(doc)
    obs, dropped = _verify(doc, [_obs(s1, s2 - 1, ["21 CFR 211.100"]), _obs(s2, s2 + 1)])
    assert [d["citation"] for d in dropped] == ["21 CFR 211.100"]
    assert obs[0].regulations == ["21 CFR 211.192"]
    assert obs[1].regulations == ["21 CFR 211.68"]


def test_chunks_respect_observation_boundaries(doc):
    s1, s2 = _numbered_starts(doc)
    obs, _ = _verify(doc, [_obs(s1, s2 - 1), _obs(s2, s2 + 1)])
    chunks = chunk(doc, obs, ["o1", "o2"])
    by_obs = {c.observation_id: c for c in chunks if c.observation_id}
    assert set(by_obs) == {"o1", "o2"}
    assert by_obs["o1"].block_end < by_obs["o2"].block_start
    assert all(c.section for c in chunks)
    assert chunks[-1].section == "Footnotes"


def test_fix_spans_claims_text_between_numbered_violations(doc):
    s1, s2 = _numbered_starts(doc)
    fixed = _fix_spans([_obs(s1, s1), _obs(s2, s2)], doc)
    assert fixed[0].last_block == s2 - 1  # the evidence paragraph and list after "1." belong to it


def test_verify_numbers_from_text(doc):
    s1, s2 = _numbered_starts(doc)
    obs, _ = _verify(doc, [_obs(s1, s2 - 1), _obs(s2, s2 + 1)])
    assert [o.number for o in obs] == [1, 2]


def test_verify_rejects_unsupported_repeat_claim(doc):
    s1, s2 = _numbered_starts(doc)
    o = _obs(s1, s2 - 1).model_copy(update={"is_repeat": True})
    obs, dropped = _verify(doc, [o])
    assert obs[0].is_repeat is False
    assert {"observation": "t", "claim": "is_repeat"} in dropped
