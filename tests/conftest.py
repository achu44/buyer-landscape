"""Shared pytest configuration.

`ALLOW_MODEL_REQUESTS = False` makes any real provider call fail loudly. Every
agent test overrides the model, so a test that reaches a provider has a bug —
better a hard error than a quiet network call against a real API key.
"""

from pathlib import Path

from pydantic_ai import models
import pytest

import deps

models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the shared retry policy but drop its real sleeps, so a test that
    exercises the retry budget runs in milliseconds. Opt-in, so a test that
    never retries still runs under the real policy."""
    monkeypatch.setattr(deps, "BACKOFF_MULTIPLIER", 0)


@pytest.fixture(autouse=True)
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every run writes a record of itself. Pointing the default at a temp
    directory for every test means no test can leave files in the repo's
    `runs/`, and a test that asserts on a record reads it from here."""
    path = tmp_path / "runs"
    monkeypatch.setattr(deps, "RUNS_DIR", str(path))
    return path
