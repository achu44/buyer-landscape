"""Fakes shared across test files. Not a test module: pytest collects only
`test_*.py`, and `pythonpath = ["."]` makes this importable as `tests.fakes`."""

from pydantic_ai import ModelResponse
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo

MODEL_NAME = "claude-test"


class ProviderDown:
    """A FunctionModel function for a provider that answers every request
    with the same HTTP error, counting how many requests it was sent."""

    def __init__(self, status_code: int = 503):
        self.status_code = status_code
        self.calls = 0

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        raise ModelHTTPError(status_code=self.status_code, model_name=MODEL_NAME)
