import pytest

from app.ingest.citations import canonical, find_citations, parent


@pytest.mark.parametrize("text,expected", [
    ("(21 CFR 211.22(a) & 211.22(d))", {"21 CFR 211.22"}),
    ("(21 CFR 211.192)", {"21 CFR 211.192"}),
    ("CGMP regulations (21 CFR, parts 210 and 211)", {"21 CFR Part 210", "21 CFR Part 211"}),
    ("the FSVP regulation in 21 CFR part 1, subpart L", {"21 CFR Part 1"}),
    ("as required by 21 CFR 1.502(a) and 1.504", {"21 CFR 1.502", "21 CFR 1.504"}),
    ("section 501(a)(2)(B) of the FD&C Act, 21 U.S.C. 351(a)(2)(B)", {"FD&C Act 501(a)"}),
    ("sections 502(a), 502(bb), and 201(n) of the FD&C Act",
     {"FD&C Act 502(a)", "FD&C Act 502(bb)", "FD&C Act 201(n)"}),
    ("FD&C Act sections 512(a) and 502(f) (21 U.S.C. §§ 360b(a), 352(f))",
     {"FD&C Act 512(a)", "FD&C Act 502(f)"}),
    ("Sections 512, 571, and 572 of the FD&C Act", {"FD&C Act 512", "FD&C Act 571", "FD&C Act 572"}),
    ("section 503B of the Act, are not", {"FD&C Act 503B"}),
    ("section 403(w) (if applicable) of the FD&C Act", {"FD&C Act 403(w)"}),
    ("21 U.S.C. 342(a)(4) only", set()),
])
def test_find_citations(text, expected):
    assert find_citations(text) == expected


@pytest.mark.parametrize("raw,expected", [
    ("21 CFR 211.192(b)", "21 CFR 211.192"),
    ("FD&C Act 501(a)(2)(B)", "FD&C Act 501(a)"),
    ("21 CFR Part 117", "21 CFR Part 117"),
    ("Section 402(a)(4)", "FD&C Act 402(a)"),
    ("21 U.S.C. § 343(k)", None),
    ("ISO 9001", None),
])
def test_canonical(raw, expected):
    assert canonical(raw) == expected


def test_parent():
    assert parent("21 CFR 211.192") == "21 CFR Part 211"
    assert parent("FD&C Act 501(a)") == "FD&C Act 501"
    assert parent("21 CFR Part 1") is None
