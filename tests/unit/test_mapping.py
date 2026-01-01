"""Mapping reshapes and nothing else — asserted both by behaviour and by imports.

The import assertion is the architectural half. `mapping.py` may not reach for grounding,
validation or amount parsing, because if it did, the value it stored would be a judgement
somebody had already made and replaying a decision would re-check nothing. The domain's
own purity test guards the same boundary from the other side.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from garnishment.adapters.llm.mapping import (
    BIN_TO_CONFIDENCE,
    bin_to_confidence,
    malformed_extraction,
    map_llm_extraction,
)
from garnishment.adapters.llm.prompt import PROMPT_VERSION
from garnishment.adapters.llm.schema import LlmExtraction
from garnishment.domain.enums import AmountRole, DocumentType, ExtractionStatus, RuleId
from garnishment.domain.routing import route
from garnishment.domain.thresholds import (
    AUTO_MIN_FIELD_CONFIDENCE,
    ROLE_MIN_CONFIDENCE,
    below_floor,
)
from tests.fixtures.extractions import (
    S3_ORIGINAL,
    S3_RESTRICTED,
    SAMPLE_3,
    sample_3_extraction,
)
from tests.unit.test_llm_schema import WORKED_EXAMPLE

MAPPING_SOURCE = Path(
    __file__
).resolve().parents[2] / "src" / "garnishment" / "adapters" / "llm" / "mapping.py"

FORBIDDEN_IMPORTS = (
    "garnishment.domain.grounding",
    "garnishment.domain.validators",
    "garnishment.domain.amount_parsing",
    "garnishment.domain.confidence",
    "garnishment.domain.operative_amount",
    "garnishment.domain.rules",
    "garnishment.domain.routing",
)


def imported_module_names(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


@pytest.mark.parametrize("forbidden", FORBIDDEN_IMPORTS)
def test_mapping_does_no_judgement_because_it_cannot_reach_the_code_that_makes_it(
    forbidden: str,
) -> None:
    imported = imported_module_names(MAPPING_SOURCE.read_text())

    assert forbidden not in imported, (
        f"mapping.py imports {forbidden}. Grounding, validation, amount parsing and "
        f"confidence arithmetic belong inside route(), recomputed from the stored "
        f"extraction on every run. A value judged here and only stored would make replay "
        f"re-read a number this adapter had already decided."
    )


def test_the_worked_example_maps_to_the_sample_3_extraction() -> None:
    """The §7.1 payload and the golden Sample 3 fixture describe the same reading.

    Quotes differ only in the whitespace the model happened to copy, which the fixture
    collapses; nothing that routing reads differs, so both route identically.
    """
    mapped = map_llm_extraction(
        LlmExtraction.model_validate(WORKED_EXAMPLE),
        letter_text=SAMPLE_3,
        model_id="claude-sonnet-5",
    )
    golden = sample_3_extraction(candidates=(S3_ORIGINAL, S3_RESTRICTED))

    assert mapped.document_type is DocumentType.RESTRICTION_ORDER
    assert mapped.is_seizure_order is True
    assert mapped.is_seizure_order_confidence == BIN_TO_CONFIDENCE["certain"]
    assert mapped.debtor_name.value == golden.debtor_name.value
    assert mapped.case_reference.value == golden.case_reference.value
    assert mapped.account_iban.value == golden.account_iban.value
    assert mapped.amount_candidates == golden.amount_candidates
    assert route(mapped).route == route(golden).route
    assert route(mapped).decisive_rule_id == route(golden).decisive_rule_id


def test_the_figure_passes_through_as_the_string_the_model_wrote() -> None:
    mapped = map_llm_extraction(
        LlmExtraction.model_validate(WORKED_EXAMPLE),
        letter_text=SAMPLE_3,
        model_id="claude-sonnet-5",
    )

    assert [c.value_as_written for c in mapped.amount_candidates] == ["4,250.00", "1,267.00"]
    assert all(isinstance(c.value_as_written, str) for c in mapped.amount_candidates)


def test_role_labels_survive_as_domain_enum_members() -> None:
    mapped = map_llm_extraction(
        LlmExtraction.model_validate(WORKED_EXAMPLE),
        letter_text=SAMPLE_3,
        model_id="claude-sonnet-5",
    )

    assert [c.role for c in mapped.amount_candidates] == [
        AmountRole.ORIGINAL,
        AmountRole.RESTRICTED,
    ]


def test_an_unreadable_figure_is_carried_rather_than_dropped() -> None:
    """Prose where a number was expected is data for the rule table, not a mapping error."""
    payload = dict(WORKED_EXAMPLE)
    payload["amount_candidates"] = [
        {
            "raw_text": "in the region of four thousand two hundred euros",
            "value_as_written": "four thousand two hundred",
            "currency": "EUR",
            "role": "operative",
            "role_confidence": "uncertain",
            "is_approximate": True,
            "evidence_quote": "in the region of four thousand two hundred euros",
            "confidence": "uncertain",
        }
    ]

    mapped = map_llm_extraction(
        LlmExtraction.model_validate(payload), letter_text=SAMPLE_3, model_id="m"
    )

    assert mapped.amount_candidates[0].value_as_written == "four thousand two hundred"
    assert mapped.amount_candidates[0].is_approximate is True


def test_a_null_field_maps_to_an_absent_value_with_its_self_report_intact() -> None:
    payload = dict(WORKED_EXAMPLE)
    payload["account_iban"] = {"value": None, "evidence_quote": None, "confidence": "guess"}

    mapped = map_llm_extraction(
        LlmExtraction.model_validate(payload), letter_text=SAMPLE_3, model_id="m"
    )

    assert mapped.account_iban.value is None
    assert mapped.account_iban.evidence_quote is None
    assert mapped.account_iban.llm_confidence == BIN_TO_CONFIDENCE["guess"]


def test_an_invalid_iban_is_wrapped_unexamined() -> None:
    """Validation is the rule table's job; mapping must not pre-empt a named rule."""
    payload = dict(WORKED_EXAMPLE)
    payload["account_iban"] = {
        "value": "NOT-AN-IBAN",
        "evidence_quote": "Account (IBAN): NOT-AN-IBAN",
        "confidence": "certain",
    }

    mapped = map_llm_extraction(
        LlmExtraction.model_validate(payload), letter_text=SAMPLE_3, model_id="m"
    )

    assert mapped.account_iban.value is not None
    assert mapped.account_iban.value.raw == "NOT-AN-IBAN"


def test_provenance_is_recorded_on_the_extraction() -> None:
    mapped = map_llm_extraction(
        LlmExtraction.model_validate(WORKED_EXAMPLE),
        letter_text=SAMPLE_3,
        model_id="claude-sonnet-5",
    )

    assert mapped.model_id == "claude-sonnet-5"
    assert mapped.prompt_version == PROMPT_VERSION
    assert mapped.extraction_status is ExtractionStatus.OK
    assert mapped.letter_text == SAMPLE_3


def test_a_malformed_reading_carries_the_letter_and_nothing_else() -> None:
    extraction = malformed_extraction(letter_text=SAMPLE_3, model_id="claude-sonnet-5")

    assert extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert extraction.letter_text == SAMPLE_3
    assert extraction.amount_candidates == ()
    assert extraction.debtor_name.value is None
    assert extraction.case_reference.value is None
    assert extraction.is_seizure_order is False


# --- the bin-to-float seam ----------------------------------------------------------------


def test_the_four_bins_are_the_whole_table() -> None:
    assert set(BIN_TO_CONFIDENCE) == {"certain", "probable", "uncertain", "guess"}


@pytest.mark.parametrize(
    ("bin_value", "expected"),
    [("certain", 0.97), ("probable", 0.88), ("uncertain", 0.60), ("guess", 0.30)],
)
def test_each_bin_maps_to_its_fixed_float(bin_value: str, expected: float) -> None:
    """Pinned, because these are the numbers the rule table's floors are compared against.

    Changing one is a routing change and should read as one in a diff, not slip through as
    a tidy-up.
    """
    assert bin_to_confidence(bin_value) == expected
    assert BIN_TO_CONFIDENCE[bin_value] == expected


def test_the_bins_sit_where_the_thresholds_need_them() -> None:
    """The spread, not any single value, is what keeps `min()` behaving sensibly.

    `certain` and `probable` clear the 0.85 floors; `uncertain` and `guess` do not, so a
    stated doubt escalates on its own. `probable` is deliberately above 0.85 rather than
    on it: landing a bin exactly on a floor would rebuild the float-equality knife edge
    the bins exist to remove.
    """
    assert BIN_TO_CONFIDENCE["probable"] > AUTO_MIN_FIELD_CONFIDENCE
    assert BIN_TO_CONFIDENCE["certain"] > BIN_TO_CONFIDENCE["probable"]
    assert not below_floor(BIN_TO_CONFIDENCE["probable"], AUTO_MIN_FIELD_CONFIDENCE)
    assert below_floor(BIN_TO_CONFIDENCE["uncertain"], AUTO_MIN_FIELD_CONFIDENCE)
    assert below_floor(BIN_TO_CONFIDENCE["uncertain"], ROLE_MIN_CONFIDENCE)
    assert below_floor(BIN_TO_CONFIDENCE["guess"], AUTO_MIN_FIELD_CONFIDENCE)
    assert BIN_TO_CONFIDENCE["guess"] < BIN_TO_CONFIDENCE["uncertain"]
    assert BIN_TO_CONFIDENCE["guess"] > 0.0


def test_a_bin_valued_payload_reaches_the_domain_as_floats() -> None:
    """The bins are wire-level only: nothing the domain touches carries a word.

    `LetterExtraction` is still typed in floats and every threshold still compares one, so
    the schema change is invisible from `route()` downwards.
    """
    mapped = map_llm_extraction(
        LlmExtraction.model_validate(WORKED_EXAMPLE), letter_text=SAMPLE_3, model_id="m"
    )

    reported = [
        mapped.is_seizure_order_confidence,
        mapped.debtor_name.llm_confidence,
        mapped.case_reference.llm_confidence,
        mapped.account_iban.llm_confidence,
        *(c.llm_confidence for c in mapped.amount_candidates),
        *(c.role_confidence for c in mapped.amount_candidates),
    ]

    assert all(isinstance(value, float) for value in reported)
    assert set(reported) <= set(BIN_TO_CONFIDENCE.values())
    assert mapped.is_seizure_order_confidence == 0.97
    assert mapped.amount_candidates[0].role_confidence == 0.88
    assert mapped.amount_candidates[1].role_confidence == 0.97


def test_a_doubtful_self_report_arrives_as_the_float_that_escalates() -> None:
    payload = dict(WORKED_EXAMPLE)
    payload["document_is_seizure_order_confidence"] = "uncertain"

    mapped = map_llm_extraction(
        LlmExtraction.model_validate(payload), letter_text=SAMPLE_3, model_id="m"
    )

    assert mapped.is_seizure_order_confidence == 0.60
    assert route(mapped).decisive_rule_id is RuleId.H02_DOC_TYPE_UNCERTAIN
