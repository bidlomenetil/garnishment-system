"""The model's only output surface: one forced tool call with this exact shape.

The schema asks the model to do reading comprehension, never legal reasoning. It never
asks "which figure is currently effective?" — that judgement is made by a versioned
doctrine in `domain.operative_amount`, where it is unit-tested and recorded. Instead the
model enumerates *every* monetary figure and labels the part each one plays, which is a
job it can do reliably and which we can check against the letter.

Validation here is schema validation and nothing else: is this a well-formed answer? It
says nothing about whether the answer is *right*. Every judgement about correctness —
grounding, checksums, format, confidence arithmetic — happens inside `route()`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EXTRACTION_TOOL_NAME = "record_garnishment_extraction"

EXTRACTION_TOOL_DESCRIPTION = (
    "Record everything you read in the garnishment letter, with a verbatim evidence "
    "quote and a self-reported confidence for every field. Call this tool exactly once."
)

ConfidenceBin = Literal["certain", "probable", "uncertain", "guess"]
"""The whole vocabulary a self-report may use. Four words, and nothing between them.

This used to be `float` bounded to [0, 1], and that was false precision. DESIGN.md §11.7
records the evidence: across a 40-call live pass, `is_seizure_order_confidence` was the
only volatile signal on otherwise-stable letters — the same Sample 3 read `true` every
time while its confidence wandered between 0.75 and 0.92, which is either side of the
0.85 floor `H02` compares it against. The *value* never moved; only the number attached
to it did. A model asked for two decimal places will produce two decimal places, but it
has no 0.01 resolution in how sure it is, so the low-order digits were noise sampled
straight onto a knife-edge threshold.

Bins remove the noise at its source rather than by widening the threshold: the model
cannot express 0.84 versus 0.86 because the schema gives it no way to say either. The
float the domain compares is then ours, fixed per bin in `mapping.BIN_TO_CONFIDENCE`,
and it moves only when we change it. Nothing downstream of that mapping knows this
happened — `route()` still reads floats, against the same thresholds, for the same
reasons."""

_BIN_MEANINGS = (
    "certain: the letter states this unambiguously and you copied it directly. "
    "probable: you are fairly sure, but a misreading is possible. "
    "uncertain: there is a real, specific reason to doubt this reading. "
    "guess: you are inferring or completing this rather than reading it."
)

CONFIDENCE_BIN_DESCRIPTION = f"How sure you are, as one of four words. {_BIN_MEANINGS}"


class _Strict(BaseModel):
    """Every confidence below is one of four words: anything else — a number, a synonym,
    a hedge — is a malformed answer rather than a very confident or a very doubtful one."""

    model_config = ConfigDict(extra="forbid")


class ExtractedFieldModel(_Strict):
    """One value the model claims to have found, with its citation and self-report."""

    value: str | None = Field(description="The value exactly as it appears; null if absent.")
    evidence_quote: str | None = Field(
        default=None,
        description="A span copied verbatim from the letter that contains this value.",
    )
    confidence: ConfidenceBin = Field(description=CONFIDENCE_BIN_DESCRIPTION)


class DebtorNameModel(ExtractedFieldModel):
    is_partial_or_abbreviated: bool = Field(
        default=False,
        description=(
            "True when the name is initialised, truncated or otherwise insufficient to "
            'identify one person, e.g. "M. Meyer".'
        ),
    )


class AmountCandidateModel(_Strict):
    """One monetary figure found in the letter, and the part the model thinks it plays."""

    raw_text: str = Field(description='The full figure as written, e.g. "EUR 4,250.00".')
    value_as_written: str = Field(
        description=(
            'The numeric part verbatim, e.g. "4,250.00". Never normalise, never convert '
            "and never compute: copy the characters. We parse it, you do not."
        )
    )
    currency: str = Field(description='Currency as written, e.g. "EUR".')
    role: Literal["operative", "original", "restricted", "unknown"] = Field(
        description=(
            "operative: the figure this letter directs to be seized. "
            "original: a figure from a prior order that this letter refers back to. "
            "restricted: a figure this letter reduces a prior order to. "
            "unknown: you cannot tell."
        )
    )
    role_confidence: ConfidenceBin = Field(
        description="How sure you are of the role label specifically. " + _BIN_MEANINGS
    )
    is_approximate: bool = Field(
        description='True if the letter hedges the figure ("approximately", "in the region of").'
    )
    evidence_quote: str | None = Field(
        default=None, description="A span copied verbatim from the letter containing this figure."
    )
    confidence: ConfidenceBin = Field(description=CONFIDENCE_BIN_DESCRIPTION)


class LlmExtraction(_Strict):
    """The complete tool-call payload."""

    document_is_seizure_order: bool = Field(
        description=(
            "True for any court order directing a bank to attach, hold or seize funds in "
            "a named account, INCLUDING a letter that restricts, reduces or otherwise "
            "amends a seizure order issued earlier — such a letter is still part of the "
            "same seizure action, not a separate, non-seizure document. False only for "
            "something outside this family entirely (a general notice, an unrelated "
            "court communication, correspondence with no attachment/seizure content)."
        )
    )
    document_is_seizure_order_confidence: ConfidenceBin = Field(
        description=CONFIDENCE_BIN_DESCRIPTION
    )
    document_type: Literal["attachment_order", "restriction_order", "other"] = Field(
        description=(
            "attachment_order: the order originating the seizure. restriction_order: a "
            "later order that restricts, reduces or amends a seizure ordered earlier — "
            "this is still document_is_seizure_order=true. other: not a seizure-related "
            "court order at all."
        )
    )

    debtor_name: DebtorNameModel
    case_reference: ExtractedFieldModel
    account_iban: ExtractedFieldModel

    amount_candidates: list[AmountCandidateModel] = Field(
        description=(
            "EVERY monetary figure in the letter, in the order they appear. Do not choose "
            "between them and do not omit one because you think it is superseded."
        )
    )

    notes: str = Field(
        default="",
        description="Anything else worth recording. Never read by the decision rules.",
    )


def extraction_tool_schema() -> dict[str, Any]:
    """The JSON schema of the single tool the model is forced to call."""
    return LlmExtraction.model_json_schema()


def extraction_tool() -> dict[str, Any]:
    """The tool definition, ready to hand to the Messages API."""
    return {
        "name": EXTRACTION_TOOL_NAME,
        "description": EXTRACTION_TOOL_DESCRIPTION,
        "input_schema": extraction_tool_schema(),
    }
