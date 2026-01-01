"""Per-field confidence, derived inside the domain from the stored extraction.

    final = min(llm_self_report, grounding_score, validator_score)

Three inputs of very different quality, combined under an operator that lets the
unverifiable one only ever lower a score. The self-report is the weakest input and cannot
be audited; it is kept because it is the only channel carrying doubt about *reading*,
which the validators — which check form — cannot see. Under `min()` its failure direction
is safe: it can escalate, never approve.

`field_confidences` is an **output** of routing, persisted as evidence for a colleague
reading the record later. It is never fed back in as an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from garnishment.domain.amount_parsing import parse_eur_amount
from garnishment.domain.extraction import AmountCandidate, LetterExtraction, candidate_at
from garnishment.domain.grounding import grounding_score, normaliser_for
from garnishment.domain.operative_amount import resolve_operative_amount
from garnishment.domain.validators import (
    ValidatorResult,
    validate_amount,
    validate_case_reference,
    validate_debtor_name,
    validate_iban,
)

DEBTOR_NAME = "debtor_name"
CASE_REFERENCE = "case_reference"
AMOUNT = "amount"
ACCOUNT_IBAN = "account_iban"

REQUIRED_FIELDS = (DEBTOR_NAME, CASE_REFERENCE, AMOUNT)
"""Fields whose confidence always gates `auto`. `account_iban` joins them when present."""


@dataclass(frozen=True, slots=True)
class FieldConfidence:
    """The four-part breakdown stored against every decision.

    `final` is a property rather than a stored number because `min()` *is* its definition;
    keeping it computed means the record can never carry a total that disagrees with its
    own parts.
    """

    llm: float
    grounding: float
    validator: float
    value_present: bool = True
    grounded_on_value_only: bool = False
    validator_code: str = ""

    @property
    def final(self) -> float:
        return min(self.llm, self.grounding, self.validator)


def derive_field_confidences(extraction: LetterExtraction) -> dict[str, FieldConfidence]:
    """Score every gated field from the extraction alone.

    Takes exactly one argument on purpose: everything it needs — the letter text, the
    evidence quotes, the self-reports — is already in the record, so replaying a decision
    genuinely re-derives these numbers instead of re-reading a conclusion an adapter drew.
    """
    resolution = resolve_operative_amount(extraction.amount_candidates, extraction.document_type)
    considered = candidate_at(extraction, resolution.considered_candidate_index)

    debtor_raw = None if extraction.debtor_name.value is None else extraction.debtor_name.value.raw
    case_raw = (
        None if extraction.case_reference.value is None else extraction.case_reference.value.raw
    )
    iban_raw = None if extraction.account_iban.value is None else extraction.account_iban.value.raw

    return {
        DEBTOR_NAME: _score(
            field_name=DEBTOR_NAME,
            letter_text=extraction.letter_text,
            llm=extraction.debtor_name.llm_confidence,
            quote=extraction.debtor_name.quote_text,
            value_text=debtor_raw,
            validator=validate_debtor_name(debtor_raw),
        ),
        CASE_REFERENCE: _score(
            field_name=CASE_REFERENCE,
            letter_text=extraction.letter_text,
            llm=extraction.case_reference.llm_confidence,
            quote=extraction.case_reference.quote_text,
            value_text=case_raw,
            validator=validate_case_reference(case_raw),
        ),
        AMOUNT: _score_amount(extraction, considered),
        ACCOUNT_IBAN: _score(
            field_name=ACCOUNT_IBAN,
            letter_text=extraction.letter_text,
            llm=extraction.account_iban.llm_confidence,
            quote=extraction.account_iban.quote_text,
            value_text=iban_raw,
            validator=validate_iban(iban_raw),
        ),
    }


def gated_field_names(extraction: LetterExtraction) -> tuple[str, ...]:
    """Fields whose confidence must clear the floor before `auto` is available.

    A *present* IBAN is load-bearing: it is the account the money leaves, and a fabricated
    one can pass mod-97 by chance, so the checksum alone must not clear it. An *absent*
    IBAN is handled by H06 and never joins this set — the brief forbids treating absence
    as invalidity.
    """
    if extraction.account_iban.value is None:
        return REQUIRED_FIELDS
    return (*REQUIRED_FIELDS, ACCOUNT_IBAN)


def _score_amount(
    extraction: LetterExtraction, considered: AmountCandidate | None
) -> FieldConfidence:
    """Score the figure the letter pointed at, parsed or not.

    Scoring the *considered* candidate rather than the resolved amount means a figure that
    never parsed is still checked for fabrication. When no candidate was identified at all
    there is no reading to be confident about, and the field scores zero as absent — which
    keeps H08's "not grounded" reason off a record where it would simply be untrue.
    """
    if considered is None:
        return FieldConfidence(
            llm=0.0,
            grounding=0.0,
            validator=0.0,
            value_present=False,
            validator_code="AMOUNT_ABSENT",
        )
    parsed: Decimal | None = parse_eur_amount(considered.value_as_written).value
    return _score(
        field_name=AMOUNT,
        letter_text=extraction.letter_text,
        llm=considered.llm_confidence,
        quote=considered.quote_text,
        value_text=considered.value_as_written,
        validator=validate_amount(parsed, considered.currency),
    )


def _score(
    *,
    field_name: str,
    letter_text: str,
    llm: float,
    quote: str | None,
    value_text: str | None,
    validator: ValidatorResult,
) -> FieldConfidence:
    grounding = grounding_score(
        letter_text=letter_text,
        evidence_quote=quote,
        value_text=value_text,
        normalise=normaliser_for(field_name),
    )
    return FieldConfidence(
        llm=llm,
        grounding=grounding.score,
        validator=validator.score,
        value_present=value_text is not None,
        grounded_on_value_only=grounding.grounded_on_value_only,
        validator_code=validator.code,
    )
