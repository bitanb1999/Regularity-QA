"""Offline tests for the graph tool's validation and for rank fusion (no databases needed)."""
import pytest

from app.graph.queries import AGGREGATE_TEMPLATES, TEMPLATES, GraphCall, InvalidGraphCall
from app.retrieval.hybrid import SECTION, Hit, rrf


@pytest.mark.parametrize("call,error", [
    (GraphCall("drop_everything"), "unknown template"),
    (GraphCall("observations_by_topic"), "requires topic"),
    (GraphCall("observations_by_topic", topic="'; MATCH (n) DETACH DELETE n //"), "unknown topic"),
    (GraphCall("observations_by_regulation", regulation="DROP DATABASE neo4j"), "unrecognized regulation"),
    (GraphCall("observations_by_company", company="   "), "requires company"),
])
def test_invalid_calls_are_rejected(call, error):
    with pytest.raises(InvalidGraphCall, match=error):
        call.validated()


def test_regulation_is_canonicalized_and_limit_clamped():
    call = GraphCall("observations_by_regulation", regulation="21 CFR 211.192(b)", limit=10_000).validated()
    assert call.regulation == "21 CFR 211.192" and call.limit == 50


def test_cypher_is_built_only_from_constants():
    # User-supplied values must never appear in the query text, only in bound parameters.
    evil = "x' }) DETACH DELETE n //"
    for template in TEMPLATES:
        call = GraphCall(template, topic="data_integrity", regulation="21 CFR Part 211", company=evil,
                         start_date="2026-01-01", end_date="2026-12-31").validated()
        assert evil not in call.cypher()
        assert call.params()["company"] == evil.strip()
        assert "DELETE" not in call.cypher().upper() and "MERGE" not in call.cypher().upper()


def test_aggregate_templates_are_known():
    assert AGGREGATE_TEMPLATES <= set(TEMPLATES)


def _hit(cid: str) -> Hit:
    return Hit(cid, "d", None, None, None, "s", None, "t", {})


def test_rrf_rewards_agreement_and_weights():
    a, b, c = _hit("a"), _hit("b"), _hit("c")
    fused = rrf([a, b], [b, c])
    assert fused[0].chunk_id == "b"  # in both lists
    weighted = rrf([c], [a, b], weights=[3.0, 1.0])
    assert weighted[0].chunk_id == "c"


def test_section_pattern():
    assert SECTION.findall("Which firms violated 21 CFR 211.192 or 1.502(a)?") == ["211.192", "1.502"]
    assert SECTION.findall("21 CFR Part 211") == []
