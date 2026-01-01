"""The fake reproduces the golden extractions exactly, and both failure modes on demand.

Equality against the fixtures is the assertion that matters. The golden cases in
`tests/fixtures/extractions.py` are what the rule table is tested against; if the fake
drifted from them by a single confidence, the API tests would be asserting one reading
while the routing tests asserted another, and the two halves of the suite would stop
agreeing without anything going red.
"""

from __future__ import annotations

import pytest

from garnishment.adapters.llm.fake_extractor import (
    PROVIDER,
    RECORDED_PAYLOADS,
    SAMPLE_1_PAYLOAD,
    FakeExtractor,
)
from garnishment.adapters.llm.mapping import BIN_TO_CONFIDENCE
from garnishment.application.errors import ExtractorUnavailable
from garnishment.domain.enums import ExtractionStatus, Route, RuleId
from garnishment.domain.routing import route
from tests.fixtures.extractions import (
    GOLDEN_CASES,
    S3_ORIGINAL,
    S3_RESTRICTED,
    SAMPLE_1,
    SAMPLE_2,
    SAMPLE_3,
    sample_3_extraction,
)

GOLDEN_BY_NAME = {case.name: case for case in GOLDEN_CASES}


@pytest.mark.parametrize(
    ("letter", "golden_case_name"),
    [
        (SAMPLE_1, "sample_1_attachment_order_auto"),
        (SAMPLE_2, "sample_2_partial_name_needs_human"),
        (SAMPLE_3, "sample_3_restriction_order_auto"),
    ],
)
async def test_the_canned_reading_equals_the_golden_extraction(
    letter: str, golden_case_name: str
) -> None:
    result = await FakeExtractor().extract(letter)

    assert result.extraction == GOLDEN_BY_NAME[golden_case_name].extraction


@pytest.mark.parametrize(
    ("letter", "expected_route", "expected_decisive"),
    [
        (SAMPLE_1, Route.AUTO, RuleId.A00_AUTO_DEFAULT),
        (SAMPLE_2, Route.NEEDS_HUMAN, RuleId.H05_DEBTOR_NAME_PARTIAL),
        (SAMPLE_3, Route.AUTO, RuleId.A00_AUTO_DEFAULT),
    ],
)
async def test_the_three_samples_route_as_labelled(
    letter: str, expected_route: Route, expected_decisive: RuleId
) -> None:
    outcome = route((await FakeExtractor().extract(letter)).extraction)

    assert outcome.route is expected_route
    assert outcome.decisive_rule_id is expected_decisive


async def test_the_restriction_order_carries_both_figures_with_their_roles() -> None:
    """The reader enumerates; the doctrine decides. Both figures must survive the call."""
    result = await FakeExtractor().extract(SAMPLE_3)

    assert result.extraction.amount_candidates == (S3_ORIGINAL, S3_RESTRICTED)
    assert result.extraction == sample_3_extraction(candidates=(S3_ORIGINAL, S3_RESTRICTED))


async def test_the_raw_payload_is_handed_back_for_the_record() -> None:
    result = await FakeExtractor().extract(SAMPLE_1)

    assert result.raw_llm_response == SAMPLE_1_PAYLOAD
    assert result.provider == PROVIDER


async def test_an_injected_malformed_reading_escalates_and_never_rejects() -> None:
    result = await FakeExtractor(malformed=True).extract(SAMPLE_1)
    outcome = route(result.extraction)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert outcome.route is Route.NEEDS_HUMAN
    assert outcome.decisive_rule_id is RuleId.H01_EXTRACTION_FAILED


async def test_an_injected_outage_raises_rather_than_returning_a_reading() -> None:
    with pytest.raises(ExtractorUnavailable):
        await FakeExtractor(unavailable=True).extract(SAMPLE_1)


async def test_a_letter_with_no_recorded_reading_escalates() -> None:
    """We have never read this letter, so the honest answer is that we could not."""
    result = await FakeExtractor().extract("A letter nobody recorded a reading for.\n")

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert route(result.extraction).route is Route.NEEDS_HUMAN


async def test_an_injected_response_overrides_the_recorded_one() -> None:
    payload = dict(SAMPLE_1_PAYLOAD)
    payload["document_is_seizure_order_confidence"] = "guess"

    result = await FakeExtractor(responses={SAMPLE_1: payload}).extract(SAMPLE_1)

    assert result.extraction.is_seizure_order_confidence == BIN_TO_CONFIDENCE["guess"]
    assert route(result.extraction).decisive_rule_id is RuleId.H02_DOC_TYPE_UNCERTAIN


async def test_the_reading_is_the_same_every_time() -> None:
    extractor = FakeExtractor()

    first = await extractor.extract(SAMPLE_3)
    second = await extractor.extract(SAMPLE_3)

    assert first == second


async def test_the_canned_payloads_answer_in_bins_and_the_reading_comes_back_in_floats() -> None:
    """The fake goes through the real schema, so it is also the proof the seam holds.

    If a payload still carried a float the Pydantic model would reject it outright; what
    this asserts is the other half — that nothing word-shaped survives into the extraction
    the rule table reads.
    """
    for payload in RECORDED_PAYLOADS.values():
        assert payload["document_is_seizure_order_confidence"] in BIN_TO_CONFIDENCE

    extraction = (await FakeExtractor().extract(SAMPLE_1)).extraction

    assert isinstance(extraction.is_seizure_order_confidence, float)
    assert isinstance(extraction.debtor_name.llm_confidence, float)
    assert isinstance(extraction.amount_candidates[0].role_confidence, float)
