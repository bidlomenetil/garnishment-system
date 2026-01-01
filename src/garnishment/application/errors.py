"""Failures the application layer raises, and what each one means about the system.

The distinction these two exceptions draw is the one the whole error design rests on:
`ExtractorUnavailable` means the reader never answered, so there is nothing to record and
the caller owes the client a 503. A reader that answered *badly* raises nothing at all —
it comes back as an extraction carrying `MALFORMED_AFTER_RETRY`, is routed, and is stored
like any other decision.
"""

from __future__ import annotations


class ExtractorUnavailable(Exception):
    """The reader could not be reached: transport, authentication or rate limiting.

    No decision record is written when this is raised. A record must record a real
    reading; writing a `needs_human` record for an outage would put an empty work item in
    a clerk's queue and make an infrastructure problem indistinguishable from an
    ambiguous letter.
    """


class DecisionNotFound(Exception):
    """No decision exists with the requested id."""

    def __init__(self, decision_id: str) -> None:
        super().__init__(f"No decision with id {decision_id!r}.")
        self.decision_id = decision_id
