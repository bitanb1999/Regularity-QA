from app.llm.answer import NOT_FOUND, REJECTED_MSG, check

AVAILABLE = {"S1", "S2", "S3"}


def test_well_cited_answer_passes():
    c = check("Curia shared passwords across analysts [S1]. Records were shredded [S2, S3].", AVAILABLE)
    assert c.status == "answered" and c.cited == ["S1", "S2", "S3"] and c.flags == []


def test_padded_and_unicode_spaced_citations_are_normalized():
    c = check("The tofu cooled too slowly for safe storage [ S3 ]. Ice was unevenly spread【S1】.", AVAILABLE)
    assert c.status == "answered"
    assert "[S3]" in c.answer and "[S1]" in c.answer and " " not in c.answer


def test_no_citations_is_rejected():
    c = check("Curia had many data integrity problems across its laboratory.", AVAILABLE)
    assert c.status == "rejected" and c.answer == REJECTED_MSG and "no_valid_citations" in c.flags


def test_only_invented_labels_is_rejected():
    c = check("Curia had many data integrity problems [S9].", AVAILABLE)
    assert c.status == "rejected" and "invalid_citations:S9" in c.flags


def test_invented_labels_are_stripped_and_flagged():
    c = check("Analysts shared passwords on lab systems [S1, S7]. Records were destroyed in bins [S8].", AVAILABLE)
    assert c.status == "flagged"
    assert "[S1]" in c.answer and "S7" not in c.answer and "S8" not in c.answer
    assert "invalid_citations:S7,S8" in c.flags


def test_mostly_uncited_answer_is_flagged():
    text = ("Curia shared passwords [S1]. The firm also destroyed records in shred bins regularly. "
            "Its quality unit failed to review audit trails for any of the systems.")
    c = check(text, AVAILABLE)
    assert c.status == "flagged" and any(f.startswith("uncited_claims") for f in c.flags)


def test_model_not_found_is_a_refusal():
    c = check(f"{NOT_FOUND} The sources do not mention Pfizer.", AVAILABLE)
    assert c.status == "refused" and c.answer == "The sources do not mention Pfizer."


def test_zero_width_characters_inside_citations():
    c = check("21 CFR Part 211 was cited in nine observations [\u200bS1] across three firms [S2\u200b].", AVAILABLE)
    assert c.status == "answered" and c.cited == ["S1", "S2"] and "\u200b" not in c.answer


def test_markdown_structure_is_not_counted_as_uncited_claims():
    text = ("**Most cited regulations in the indexed warning letters**\n\n"
            "| Rank | Regulation cited | Number of observations | Companies |\n"
            "|------|------------------|------------------------|-----------|\n"
            "| 1 | 21 CFR Part 211 | 9 | Babikian, Bentley, kdc/one [S1] |\n"
            "| 2 | FD&C Act 502 | 8 | Bentley, Stokes Healthcare [S2] |\n")
    c = check(text, AVAILABLE)
    assert c.status == "answered" and c.coverage == 1.0
