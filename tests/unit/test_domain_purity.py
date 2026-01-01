"""The domain imports nothing outside the standard library.

This is the assignment's thesis made structural rather than aspirational. The deterministic
half of the system lives in a package that cannot reach a model, a database or an HTTP
client, so "the probabilistic part ends at one boundary line" is an enforced property and
not an import convention somebody will breach in six months.

It is also what makes the rest of the guarantees cheap. A package with no I/O is testable
with no fixtures, and `route()` being a pure function of a stored record is what lets a
decision be replayed years later.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

DOMAIN_ROOT = Path(__file__).resolve().parents[2] / "src" / "garnishment" / "domain"
DOMAIN_PACKAGE = "garnishment.domain"

DOMAIN_MODULES = sorted(DOMAIN_ROOT.rglob("*.py"))


def imported_module_names(source: str) -> list[str]:
    """Every module a file imports, as written, with relative imports resolved as local."""
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.append(DOMAIN_PACKAGE)
            elif node.module is not None:
                names.append(node.module)
    return names


def test_the_domain_package_is_not_empty() -> None:
    """Guards the parametrised test below against passing because it found no files."""
    assert len(DOMAIN_MODULES) >= 10


@pytest.mark.parametrize("path", DOMAIN_MODULES, ids=lambda p: p.name)
def test_domain_module_imports_only_the_standard_library(path: Path) -> None:
    offending: list[str] = []
    for name in imported_module_names(path.read_text()):
        if name == DOMAIN_PACKAGE or name.startswith(f"{DOMAIN_PACKAGE}."):
            continue
        if name.split(".")[0] in sys.stdlib_module_names:
            continue
        offending.append(name)
    assert offending == [], (
        f"{path.name} imports {offending}; the domain may use only the standard library "
        f"and other domain modules."
    )


def test_the_domain_does_not_reach_the_layers_above_it() -> None:
    forbidden = ("garnishment.adapters", "garnishment.application", "garnishment.api")
    for path in DOMAIN_MODULES:
        for name in imported_module_names(path.read_text()):
            assert not name.startswith(forbidden), f"{path.name} imports {name}"


def test_named_third_party_packages_are_absent_from_the_domain() -> None:
    """Spelled out because these are the ones a future change would reach for first."""
    banned = ("pydantic", "sqlalchemy", "fastapi", "anthropic", "httpx", "ulid")
    for path in DOMAIN_MODULES:
        names = imported_module_names(path.read_text())
        for name in names:
            assert name.split(".")[0] not in banned, f"{path.name} imports {name}"


def test_the_domain_reads_no_clock_and_no_randomness() -> None:
    """A decision must be reproducible from its stored record alone.

    `datetime` is permitted as a *type* on the audit record, but the domain must never
    read the current time or a random source: both would make a replay disagree with the
    original run for reasons that have nothing to do with the letter.
    """
    for path in DOMAIN_MODULES:
        source = path.read_text()
        for forbidden in ("datetime.now", "datetime.utcnow", "time.time", "random.", "uuid4"):
            assert forbidden not in source, f"{path.name} contains {forbidden}"
        assert "import random" not in source, path.name
