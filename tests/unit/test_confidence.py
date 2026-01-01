"""Confidence is derived, never accepted.

`min(llm_self_report, grounding, validator)` combines one unverifiable signal with two
verifiable ones under an operator that lets the unverifiable one only ever lower a score.
Nothing in the pipeline can raise a confidence.
"""

from __future__ import annotations

from garnishment.domain.confidence import (
    ACCOUNT_IBAN,
    AMOUNT,
    CASE_REFERENCE,
    DEBTOR_NAME,
    FieldConfidence,
    derive_field_confidences,
    gated_field_names,
)
from garnishment.domain.enums import AmountRole, DocumentType
from garnishment.domain.extraction import AmountCandidate, ExtractedField, LetterExtraction
from garnishment.domain.values import CaseReference, DebtorName, EvidenceQuote, Iban

LETTER = (
    "DISTRICT COURT OF EXAMPLETON\n"
    "Case reference: AB-1042/24\n"
    "Debtor:        Jordan A. Meyer\n"
    "Account (IBAN): DE44 5001 0517 5407 3249 31\n"
    "Amount to seize: EUR 4,250.00\n"
)


def well_formed(**overrides: object) -> LetterExtraction:
    defaults: dict[str, object] = {
        "letter_text": LETTER,
        "document_type": DocumentType.ATTACHMENT_ORDER,
        "is_seizure_order": True,
        "is_seizure_order_confidence": 0.98,
        "debtor_name": ExtractedField[DebtorName](
            DebtorName("Jordan A. Meyer"),
            EvidenceQuote("Debtor:        Jordan A. Meyer"),
            0.97,
        ),
        "case_reference": ExtractedField[CaseReference](
            CaseReference("AB-1042/24"), EvidenceQuote("Case reference: AB-1042/24"), 0.99
        ),
        "account_iban": ExtractedField[Iban](
            Iban("DE44 5001 0517 5407 3249 31"),
            EvidenceQuote("Account (IBAN): DE44 5001 0517 5407 3249 31"),
            0.98,
        ),
        "amount_candidates": (
            AmountCandidate(
                value_as_written="4,250.00",
                currency="EUR",
                raw_text="EUR 4,250.00",
                role=AmountRole.OPERATIVE,
                role_confidence=0.96,
                evidence_quote=EvidenceQuote("Amount to seize: EUR 4,250.00"),
                llm_confidence=0.97,
            ),
        ),
    }
    defaults.update(overrides)
    return LetterExtraction(**defaults)  # type: ignore[arg-type]


def test_final_is_the_minimum_of_the_three_components() -> None:
    assert FieldConfidence(llm=0.97, grounding=1.0, validator=1.0).final == 0.97
    assert FieldConfidence(llm=1.0, grounding=0.90, validator=1.0).final == 0.90
    assert FieldConfidence(llm=1.0, grounding=1.0, validator=0.0).final == 0.0


def test_a_zero_component_propagates_to_zero() -> None:
    for zeroed in ("llm", "grounding", "validator"):
        components = {"llm": 0.99, "grounding": 1.0, "validator": 1.0, zeroed: 0.0}
        assert FieldConfidence(**components).final == 0.0  # type: ignore[arg-type]


def test_a_high_self_report_cannot_raise_a_failing_validator() -> None:
    """The point of `min()`: the one unverifiable input can only ever escalate."""
    extraction = well_formed(
        debtor_name=ExtractedField[DebtorName](
            DebtorName("M. Meyer"), EvidenceQuote("holder M. Meyer"), 1.0
        ),
        letter_text=LETTER + "The account holder M. Meyer is the debtor.\n",
    )
    confidence = derive_field_confidences(extraction)[DEBTOR_NAME]
    assert confidence.llm == 1.0
    assert confidence.grounding == 1.0
    assert confidence.validator == 0.0
    assert confidence.final == 0.0


def test_derivation_of_a_well_formed_extraction() -> None:
    confidences = derive_field_confidences(well_formed())
    assert set(confidences) == {DEBTOR_NAME, CASE_REFERENCE, ACCOUNT_IBAN, AMOUNT}
    for name, confidence in confidences.items():
        assert confidence.grounding == 1.0, name
        assert confidence.validator == 1.0, name
        assert confidence.value_present is True, name
    assert confidences[DEBTOR_NAME].final == 0.97
    assert confidences[CASE_REFERENCE].final == 0.99
    assert confidences[ACCOUNT_IBAN].final == 0.98
    assert confidences[AMOUNT].final == 0.97


def test_the_amount_field_is_scored_against_the_considered_candidate() -> None:
    """An unparseable figure is still checked for fabrication."""
    extraction = well_formed(
        amount_candidates=(
            AmountCandidate(
                value_as_written="four thousand two hundred",
                raw_text="in the region of four thousand two hundred euros",
                role=AmountRole.OPERATIVE,
                role_confidence=0.9,
                is_approximate=True,
                evidence_quote=EvidenceQuote("Amount to seize: EUR 4,250.00"),
                llm_confidence=0.55,
            ),
        )
    )
    amount = derive_field_confidences(extraction)[AMOUNT]
    assert amount.value_present is True
    assert amount.grounding == 1.0
    assert amount.validator == 0.0
    assert amount.final == 0.0


def test_absent_values_score_zero_and_are_marked_absent() -> None:
    extraction = well_formed(
        account_iban=ExtractedField[Iban](None),
        amount_candidates=(),
    )
    confidences = derive_field_confidences(extraction)
    assert confidences[ACCOUNT_IBAN].value_present is False
    assert confidences[ACCOUNT_IBAN].final == 0.0
    assert confidences[AMOUNT].value_present is False
    assert confidences[AMOUNT].final == 0.0


def test_an_unidentifiable_operative_figure_is_not_treated_as_a_fabricated_one() -> None:
    """Two rival figures is an ambiguity, not an ungrounded value.

    Marking the amount absent here keeps H08 ("this value is not in the letter") from
    firing with a reason that would simply be untrue.
    """
    extraction = well_formed(
        amount_candidates=(
            AmountCandidate(value_as_written="4,250.00", role=AmountRole.UNKNOWN),
            AmountCandidate(value_as_written="1,267.00", role=AmountRole.UNKNOWN),
        )
    )
    amount = derive_field_confidences(extraction)[AMOUNT]
    assert amount.value_present is False
    assert amount.final == 0.0


def test_the_value_only_grounding_tier_is_reported() -> None:
    extraction = well_formed(
        debtor_name=ExtractedField[DebtorName](
            DebtorName("Jordan A. Meyer"), EvidenceQuote("the court names the debtor thus"), 0.97
        )
    )
    confidence = derive_field_confidences(extraction)[DEBTOR_NAME]
    assert confidence.grounding == 0.90
    assert confidence.grounded_on_value_only is True


def test_derivation_reads_only_the_extraction() -> None:
    """The signature is the guarantee: one argument, no clock, no I/O, no pre-computation."""
    import inspect

    parameters = inspect.signature(derive_field_confidences).parameters
    assert list(parameters) == ["extraction"]


def test_derivation_is_deterministic() -> None:
    extraction = well_formed()
    assert derive_field_confidences(extraction) == derive_field_confidences(extraction)


def test_the_iban_joins_the_gated_set_only_when_it_is_present() -> None:
    assert gated_field_names(well_formed()) == (
        DEBTOR_NAME,
        CASE_REFERENCE,
        AMOUNT,
        ACCOUNT_IBAN,
    )
    assert gated_field_names(well_formed(account_iban=ExtractedField[Iban](None))) == (
        DEBTOR_NAME,
        CASE_REFERENCE,
        AMOUNT,
    )


def test_a_candidate_index_outside_the_stored_list_resolves_to_no_candidate() -> None:
    """Defensive: an index decoded from an older record must not raise on lookup."""
    from garnishment.domain.extraction import candidate_at

    extraction = well_formed()
    assert candidate_at(extraction, None) is None
    assert candidate_at(extraction, -1) is None
    assert candidate_at(extraction, 7) is None
    assert candidate_at(extraction, 0) is extraction.amount_candidates[0]
