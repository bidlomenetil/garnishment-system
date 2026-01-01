"""Stable vocabulary for the decision boundary.

Every member of :class:`RuleId` is an audit identifier that appears in stored decision
records. Members are never renamed or reused; a retired rule keeps its identifier.
"""

from __future__ import annotations

from enum import StrEnum


class Route(StrEnum):
    """The three outcomes the service can reach.

    There is deliberately **no** severity ordering on this enum. Escalation and rejection
    are different kinds of outcome, not two points on one scale: ``reject`` says something
    about the letter, ``needs_human`` says something about our reading of it. A total order
    would invite a "severity never decreases" invariant that is false by design — lowering
    the model's confidence that a document is *not* a seizure order correctly moves it from
    ``reject`` to ``needs_human``, because escalation requires less evidence than rejection.
    """

    AUTO = "auto"
    NEEDS_HUMAN = "needs_human"
    REJECT = "reject"


class DocumentType(StrEnum):
    ATTACHMENT_ORDER = "attachment_order"
    RESTRICTION_ORDER = "restriction_order"
    OTHER = "other"


class AmountRole(StrEnum):
    """The part a monetary figure plays in the letter, as labelled by the reader."""

    OPERATIVE = "operative"
    ORIGINAL = "original"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class ResolutionReason(StrEnum):
    """Why `resolve_operative_amount` reached the answer it did."""

    NO_AMOUNT = "no_amount"
    SOLE_CANDIDATE = "sole_candidate"
    EXPLICIT_OPERATIVE = "explicit_operative"
    RESTRICTION_SUPERSEDES = "restriction_supersedes"
    RESTRICTION_UNRESOLVED = "restriction_unresolved"
    AMBIGUOUS = "ambiguous"
    UNPARSEABLE = "unparseable"


class ExtractionStatus(StrEnum):
    """Whether the reader produced a well-formed answer at all.

    ``MALFORMED_AFTER_RETRY`` is *our* failure, never the letter's. It escalates under
    ``H01`` and must never cause a rejection.
    """

    OK = "ok"
    MALFORMED_AFTER_RETRY = "malformed_after_retry"


class RuleId(StrEnum):
    """Stable audit identifiers for the ordered rule table (see ``rules.RULES``).

    Prefixes carry meaning: ``R*`` reject, ``H*`` escalate to a human, ``A00`` the
    automatic default, ``I*`` informational and never decisive.
    """

    # --- Reject: evidence that the letter itself is invalid -----------------------------
    R01_NOT_SEIZURE_ORDER = "R01_NOT_SEIZURE_ORDER"
    R02_CASE_REFERENCE_MISSING = "R02_CASE_REFERENCE_MISSING"
    R03_CASE_REFERENCE_MALFORMED = "R03_CASE_REFERENCE_MALFORMED"
    R04_AMOUNT_MISSING = "R04_AMOUNT_MISSING"
    R05_AMOUNT_NON_POSITIVE = "R05_AMOUNT_NON_POSITIVE"

    # --- Escalate: reading failure -----------------------------------------------------
    H01_EXTRACTION_FAILED = "H01_EXTRACTION_FAILED"
    H17_DOCUMENT_TYPE_CONTRADICTION = "H17_DOCUMENT_TYPE_CONTRADICTION"
    H02_DOC_TYPE_UNCERTAIN = "H02_DOC_TYPE_UNCERTAIN"

    # --- Escalate: identity, whose money is this ---------------------------------------
    H05_DEBTOR_NAME_PARTIAL = "H05_DEBTOR_NAME_PARTIAL"
    H06_IBAN_ABSENT = "H06_IBAN_ABSENT"
    H07_IBAN_INVALID = "H07_IBAN_INVALID"

    # --- Escalate: amount, how much ----------------------------------------------------
    H11_OPERATIVE_AMOUNT_UNRESOLVED = "H11_OPERATIVE_AMOUNT_UNRESOLVED"
    H12_RESTRICTION_NOT_REDUCING = "H12_RESTRICTION_NOT_REDUCING"
    H03_AMOUNT_AMBIGUOUS = "H03_AMOUNT_AMBIGUOUS"
    H18_MULTIPLE_FIGURES_NO_RESTRICTION_CONTEXT = "H18_MULTIPLE_FIGURES_NO_RESTRICTION_CONTEXT"
    H04_AMOUNT_APPROXIMATE = "H04_AMOUNT_APPROXIMATE"
    H16_AMOUNT_VALUE_UNPARSEABLE = "H16_AMOUNT_VALUE_UNPARSEABLE"
    H13_AMOUNT_FORMAT_AMBIGUOUS = "H13_AMOUNT_FORMAT_AMBIGUOUS"
    H14_CURRENCY_NOT_EUR = "H14_CURRENCY_NOT_EUR"
    H15_AMOUNT_IMPLAUSIBLE = "H15_AMOUNT_IMPLAUSIBLE"

    # --- Escalate: evidence quality ----------------------------------------------------
    H08_UNGROUNDED_EVIDENCE = "H08_UNGROUNDED_EVIDENCE"
    H09_LOW_CONFIDENCE_FIELD = "H09_LOW_CONFIDENCE_FIELD"

    # --- Automatic default -------------------------------------------------------------
    A00_AUTO_DEFAULT = "A00_AUTO_DEFAULT"

    # --- Informational: recorded, never decisive ---------------------------------------
    I01_OPERATIVE_AMOUNT_RESOLVED = "I01_OPERATIVE_AMOUNT_RESOLVED"
    I02_IBAN_VALIDATED = "I02_IBAN_VALIDATED"
    I03_GROUNDED_ON_VALUE = "I03_GROUNDED_ON_VALUE"
    I04_INJECTION_SUSPECTED = "I04_INJECTION_SUSPECTED"
