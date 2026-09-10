"""Unit tests for deps.py — the shared Deps object and its resilient HTTP
client. No real network calls: httpx.MockTransport simulates transient
failures so we can verify retry behavior deterministically."""

import asyncio
import logging

import httpx
import pytest

from deps import MAX_ATTEMPTS, REQUEST_TIMEOUT, build_deps, build_http_client

RUN_ID = "run-abc123"


def run(coro):
    return asyncio.run(coro)


def test_client_has_a_timeout_configured():
    client = build_http_client(RUN_ID)
    assert client.timeout == REQUEST_TIMEOUT


def test_deps_carries_client_and_run_id():
    deps = build_deps(RUN_ID)
    assert deps.run_id == RUN_ID
    assert isinstance(deps.http_client, httpx.AsyncClient)


def test_retries_transient_failure_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectTimeout("simulated transient timeout", request=request)
        return httpx.Response(200, json={"ok": True})

    client = build_http_client(RUN_ID, transport=httpx.MockTransport(handler))

    response = run(client.get("https://example.test/search"))

    assert response.status_code == 200
    assert calls["n"] == 3  # two failures + one success: retry actually happened


def test_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    client = build_http_client(RUN_ID, transport=httpx.MockTransport(handler))

    response = run(client.get("https://example.test/search"))

    assert response.status_code == 200
    assert calls["n"] == 2


def test_gives_up_after_max_attempts():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("always fails", request=request)

    client = build_http_client(RUN_ID, transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.ConnectTimeout):
        run(client.get("https://example.test/search"))


def test_does_not_retry_4xx():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    client = build_http_client(RUN_ID, transport=httpx.MockTransport(handler))

    response = run(client.get("https://example.test/search"))

    assert response.status_code == 404
    assert calls["n"] == 1  # not retried: a 404 is not transient


def test_retry_log_line_carries_run_id(caplog):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectTimeout("simulated transient timeout", request=request)
        return httpx.Response(200)

    client = build_http_client(RUN_ID, transport=httpx.MockTransport(handler))

    with caplog.at_level(logging.WARNING, logger="deps"):
        run(client.get("https://example.test/search"))

    retry_logs = [r for r in caplog.records if "retrying after transient failure" in r.message]
    assert retry_logs, "expected a retry log line"
    assert all(RUN_ID in r.message for r in retry_logs)
