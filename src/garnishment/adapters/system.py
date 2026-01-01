"""The two trivial adapters: real time and real identity.

They are ports rather than direct calls so that `Decision` construction is deterministic
in tests. A replayed record must differ from the original in nothing at all, not even its
identity, and that is only possible if the id and the timestamp arrive from outside.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ulid import ULID


class SystemClock:
    """Wall-clock time, always UTC. A record's timestamp carries no local offset."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class UlidGenerator:
    """ULIDs, so decision ids sort by creation time without a separate index."""

    def new_id(self) -> str:
        return str(ULID())
