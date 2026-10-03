"""Offline tests for the answer-cache key (no database: data_version is stubbed)."""
import pytest

from app import cache
from app.models.api import AskRequest


@pytest.fixture(autouse=True)
def fixed_data_version(monkeypatch):
    monkeypatch.setattr(cache, "data_version", lambda: "10:197:2026-10-02")


def test_normalization_ignores_case_spacing_and_trailing_punctuation():
    a = cache.key(AskRequest(question="What did FDA find at Curia?"))
    b = cache.key(AskRequest(question="  what did FDA   find at curia "))
    assert a == b


def test_different_companies_never_share_a_key():
    a = cache.key(AskRequest(question="What did FDA find at Bentley?"))
    b = cache.key(AskRequest(question="What did FDA find at Babikian?"))
    assert a != b


def test_retrieval_settings_are_part_of_the_key():
    base = cache.key(AskRequest(question="Which firms violated 211.192?"))
    assert base != cache.key(AskRequest(question="Which firms violated 211.192?", use_graph=False))
    assert base != cache.key(AskRequest(question="Which firms violated 211.192?", mode="vector"))


def test_reingesting_data_invalidates(monkeypatch):
    before = cache.key(AskRequest(question="What did FDA find at Curia?"))
    monkeypatch.setattr(cache, "data_version", lambda: "11:205:2026-10-04")
    assert before != cache.key(AskRequest(question="What did FDA find at Curia?"))


def test_only_checked_outcomes_are_cacheable():
    assert cache.CACHEABLE == {"answered", "flagged", "refused"}
