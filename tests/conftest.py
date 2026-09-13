"""Shared pytest configuration.

`ALLOW_MODEL_REQUESTS = False` makes any real provider call fail loudly. Every
agent test overrides the model, so a test that reaches a provider has a bug —
better a hard error than a quiet network call against a real API key.
"""

from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False
