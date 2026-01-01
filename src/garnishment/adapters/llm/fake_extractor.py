"""A deterministic reader that replays recorded tool-call payloads.

This is what makes `make test` run green with no API key, and it is not a stub: the canned
payloads go through the same Pydantic schema and the same `map_llm_extraction` as a live
response, so the tests that use it exercise the real boundary rather than a shortcut
around it. Only the transport is fake.

Payloads are selected by the SHA-256 of the letter, so a fixture cannot silently drift
away from the letter it claims to describe — edit `evals/samples/sample_01.txt` and the
lookup misses immediately rather than replaying a reading of the old bytes.

A letter with no recorded reading escalates under H01 rather than being invented for: the
honest answer to "we have never read this letter" is that we could not read it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from garnishment.adapters.llm.mapping import malformed_extraction, map_llm_extraction
from garnishment.adapters.llm.prompt import PROMPT_VERSION
from garnishment.adapters.llm.schema import LlmExtraction
from garnishment.application.errors import ExtractorUnavailable
from garnishment.application.ports import ExtractionResult

PROVIDER = "fake"

REPLAYED_MODEL_ID = "fake-claude-sonnet-5"
"""Names the reader in every replayed record: the model these payloads were recorded
from, under a prefix that says no model was called.

The prefix is the whole point. `LLM_PROVIDER` defaults to `fake`, so a deployment that
was never configured still answers; stamping the bare model id into those records would
make a misconfigured deployment's audit trail indistinguishable from a real reading months
later, when the configuration is no longer there to check."""

SAMPLE_1_SHA256 = "839d9a3312fdcc70e0a8196b190f5be9df1e8dc7dea2ed710403f095f1e0d91c"
SAMPLE_2_SHA256 = "33983f7c8080d914ef806d481a0676448eef224f7445f7acbe5bcc113736e194"
SAMPLE_3_SHA256 = "9b133c7c0d54483c76d091dbbaa99511fc6d66666db4cdd10275c99a26bf1993"

SAMPLE_1_PAYLOAD: Mapping[str, Any] = {
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "attachment_order",
    "debtor_name": {
        "value": "Jordan A. Meyer",
        "evidence_quote": "Debtor: Jordan A. Meyer",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "AB-1042/24",
        "evidence_quote": "Case reference: AB-1042/24",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN): DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    "amount_candidates": [
        {
            "raw_text": "EUR 4,250.00",
            "value_as_written": "4,250.00",
            "currency": "EUR",
            "role": "operative",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Amount to seize: EUR 4,250.00",
            "confidence": "certain",
        }
    ],
    "notes": "",
}

SAMPLE_2_PAYLOAD: Mapping[str, Any] = {
    "document_is_seizure_order": True,
    # Terse, but on court letterhead, citing a case reference and instructing attachment,
    # so `probable` rather than doubtful, which clears the doc-type floor. An `uncertain`
    # here would escalate this letter under H02 instead of H05, and the partial name — the
    # thing a clerk actually has to resolve — would stop being the decisive reason.
    "document_is_seizure_order_confidence": "probable",
    "document_type": "attachment_order",
    "debtor_name": {
        "value": "M. Meyer",
        "evidence_quote": "the account holder M. Meyer",
        "confidence": "probable",
        "is_partial_or_abbreviated": True,
    },
    "case_reference": {
        "value": "AB-1042/24",
        "evidence_quote": "Case reference: AB-1042/24",
        "confidence": "certain",
    },
    "account_iban": {"value": None, "evidence_quote": None, "confidence": "guess"},
    "amount_candidates": [
        {
            "raw_text": "in the region of four thousand two hundred euros",
            "value_as_written": "four thousand two hundred",
            "currency": "EUR",
            "role": "operative",
            "role_confidence": "uncertain",
            "is_approximate": True,
            "evidence_quote": (
                "The sum concerned is in the region of four thousand two hundred euros"
            ),
            "confidence": "uncertain",
        }
    ],
    "notes": "",
}

SAMPLE_3_PAYLOAD: Mapping[str, Any] = {
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "restriction_order",
    "debtor_name": {
        "value": "Jordan A. Meyer",
        "evidence_quote": "Debtor: Jordan A. Meyer",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "AB-1042/24",
        "evidence_quote": "Case reference: AB-1042/24",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN): DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    # Both figures, both labelled. The model is never asked which one is effective; the
    # doctrine in `domain.operative_amount` decides that, and records why.
    "amount_candidates": [
        {
            "raw_text": "EUR 4,250.00",
            "value_as_written": "4,250.00",
            "currency": "EUR",
            "role": "original",
            "role_confidence": "probable",
            "is_approximate": False,
            "evidence_quote": (
                "which directed the attachment of funds in the amount of EUR 4,250.00"
            ),
            "confidence": "certain",
        },
        {
            "raw_text": "EUR 1,267.00",
            "value_as_written": "1,267.00",
            "currency": "EUR",
            "role": "restricted",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Restricted amount to seize: EUR 1,267.00",
            "confidence": "certain",
        },
    ],
    "notes": "",
}

# --- Adversarial letters (evals/samples/extra_*.txt) -------------------------------------
#
# These exist so `make eval` exercises the reject block, the restriction-consistency check
# and the injection heuristic without a key. The readings below are what a competent reader
# should return for each letter: note that none of them invents a defect. Every route these
# letters take is decided by `route()` from an honest reading, which is the point — the
# adversarial content is in the *letters*, not in the payloads.

EXTRA_1_SHA256 = "7d1943cd9e18cdcdb3b5bd9192601d9097a11251be2f2226bc7ae0db8ce39698"
EXTRA_2_SHA256 = "3e939a52f76169a6febb2fa6db567b361126f3276ddaa9d62aa1781053ce4a21"
EXTRA_3_SHA256 = "2b0b25d9150ff654912696cab4233d7bc0d2e92cb110bb82875f0de9305c770c"
EXTRA_4_SHA256 = "14b7d18c7fb9820f55d95d8960395cf39f09ee63af0a4804d15b29be6236cc7c"
EXTRA_5_SHA256 = "ac46296a1905c0e0e2ecc7ce989155a6682db137597dcaf7aadc592f8f1f0b36"

EXTRA_1_PAYLOAD: Mapping[str, Any] = {
    # A routine bank charge notice. The reader is confident it is not a court order, which
    # is the one judgement in the table allowed to produce a rejection (R01).
    "document_is_seizure_order": False,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "other",
    "debtor_name": {
        "value": None,
        "evidence_quote": None,
        "confidence": "guess",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {"value": None, "evidence_quote": None, "confidence": "guess"},
    "account_iban": {"value": None, "evidence_quote": None, "confidence": "guess"},
    "amount_candidates": [
        {
            "raw_text": "EUR 12.50",
            "value_as_written": "12.50",
            "currency": "EUR",
            "role": "unknown",
            "role_confidence": "uncertain",
            "is_approximate": False,
            "evidence_quote": "in the amount of EUR 12.50",
            "confidence": "certain",
        }
    ],
    "notes": "Account maintenance charge, not an enforcement instruction.",
}

EXTRA_2_PAYLOAD: Mapping[str, Any] = {
    # A genuine order whose case reference is written the wrong way round. The reader copies
    # it faithfully; the malformation is the letter's, so R03 rejects.
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "attachment_order",
    "debtor_name": {
        "value": "Priya N. Raghavan",
        "evidence_quote": "Debtor: Priya N. Raghavan",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "AB/1042-24",
        "evidence_quote": "Case reference: AB/1042-24",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN): DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    "amount_candidates": [
        {
            "raw_text": "EUR 3,120.00",
            "value_as_written": "3,120.00",
            "currency": "EUR",
            "role": "operative",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Amount to seize: EUR 3,120.00",
            "confidence": "certain",
        }
    ],
    "notes": "",
}

EXTRA_3_PAYLOAD: Mapping[str, Any] = {
    # An order for nothing. The figure reads cleanly and grounds perfectly; it is simply not
    # a positive amount, which R05 treats as evidence about the letter rather than about us.
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "attachment_order",
    "debtor_name": {
        "value": "Tomas Lindqvist",
        "evidence_quote": "Debtor: Tomas Lindqvist",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "CD-0917/25",
        "evidence_quote": "Case reference: CD-0917/25",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN): DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    "amount_candidates": [
        {
            "raw_text": "EUR 0.00",
            "value_as_written": "0.00",
            "currency": "EUR",
            "role": "operative",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Amount to seize: EUR 0.00",
            "confidence": "certain",
        }
    ],
    "notes": "",
}

EXTRA_4_PAYLOAD: Mapping[str, Any] = {
    # A restriction that raises the figure it claims to reduce. Both labels are confident and
    # both figures parse, so the doctrine resolves cleanly and H12 catches the inconsistency
    # afterwards — the reader is not asked to notice the contradiction.
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "restriction_order",
    "debtor_name": {
        "value": "Jordan A. Meyer",
        "evidence_quote": "Debtor: Jordan A. Meyer",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "AB-1042/24",
        "evidence_quote": "Case reference: AB-1042/24",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN): DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    "amount_candidates": [
        {
            "raw_text": "EUR 4,250.00",
            "value_as_written": "4,250.00",
            "currency": "EUR",
            "role": "original",
            "role_confidence": "probable",
            "is_approximate": False,
            "evidence_quote": (
                "which directed the attachment of funds in the amount of EUR 4,250.00"
            ),
            "confidence": "certain",
        },
        {
            "raw_text": "EUR 5,900.00",
            "value_as_written": "5,900.00",
            "currency": "EUR",
            "role": "restricted",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Restricted amount to seize: EUR 5,900.00",
            "confidence": "certain",
        },
    ],
    "notes": "",
}

EXTRA_5_PAYLOAD: Mapping[str, Any] = {
    # A valid order carrying a paragraph addressed to the automated reader. The reading below
    # is what happens when the injection fails: the reader reports the letter's real fields
    # and nothing else. I04 records that the text was there; it does not change the route,
    # because grounding — not a substring heuristic — is the control that makes an injected
    # value impossible to smuggle into a typed field.
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "attachment_order",
    "debtor_name": {
        "value": "Hannah Okonkwo",
        "evidence_quote": "Debtor: Hannah Okonkwo",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "EF-2231/26",
        "evidence_quote": "Case reference: EF-2231/26",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN): DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    "amount_candidates": [
        {
            "raw_text": "EUR 2,480.00",
            "value_as_written": "2,480.00",
            "currency": "EUR",
            "role": "operative",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Amount to seize: EUR 2,480.00",
            "confidence": "certain",
        }
    ],
    "notes": "The letter contains a paragraph addressed to the processing system.",
}


RECORDED_PAYLOADS: Mapping[str, Mapping[str, Any]] = {
    SAMPLE_1_SHA256: SAMPLE_1_PAYLOAD,
    SAMPLE_2_SHA256: SAMPLE_2_PAYLOAD,
    SAMPLE_3_SHA256: SAMPLE_3_PAYLOAD,
    EXTRA_1_SHA256: EXTRA_1_PAYLOAD,
    EXTRA_2_SHA256: EXTRA_2_PAYLOAD,
    EXTRA_3_SHA256: EXTRA_3_PAYLOAD,
    EXTRA_4_SHA256: EXTRA_4_PAYLOAD,
    EXTRA_5_SHA256: EXTRA_5_PAYLOAD,
}


class FakeExtractor:
    """Replays recorded payloads, with both reader failure modes injectable.

    `malformed` and `unavailable` exist so the two failure paths can be driven from a
    test without a network: they are the difference between *answered badly* — recorded,
    escalated under H01 — and *never answered* — no record at all, 503.
    """

    def __init__(
        self,
        *,
        model_id: str = REPLAYED_MODEL_ID,
        responses: Mapping[str, Mapping[str, Any]] | None = None,
        malformed: bool = False,
        unavailable: bool = False,
    ) -> None:
        self._model_id = model_id
        self._responses = dict(responses or {})
        self._malformed = malformed
        self._unavailable = unavailable

    @property
    def provider(self) -> str:
        return PROVIDER

    async def extract(self, letter_text: str) -> ExtractionResult:
        if self._unavailable:
            raise ExtractorUnavailable("FakeExtractor was configured to be unreachable.")

        payload = self._payload_for(letter_text)
        if self._malformed or payload is None:
            return self._malformed_result(letter_text)

        llm = LlmExtraction.model_validate(payload)
        return ExtractionResult(
            extraction=map_llm_extraction(
                llm,
                letter_text=letter_text,
                model_id=self._model_id,
                prompt_version=PROMPT_VERSION,
            ),
            raw_llm_response=payload,
            provider=PROVIDER,
            notes=llm.notes,
        )

    def _payload_for(self, letter_text: str) -> Mapping[str, Any] | None:
        """An injected response wins over a recorded one, so a test can key on exact text."""
        if letter_text in self._responses:
            return self._responses[letter_text]
        digest = hashlib.sha256(letter_text.encode("utf-8")).hexdigest()
        return RECORDED_PAYLOADS.get(digest)

    def _malformed_result(self, letter_text: str) -> ExtractionResult:
        return ExtractionResult(
            extraction=malformed_extraction(
                letter_text=letter_text,
                model_id=self._model_id,
                prompt_version=PROMPT_VERSION,
            ),
            raw_llm_response={},
            provider=PROVIDER,
        )
