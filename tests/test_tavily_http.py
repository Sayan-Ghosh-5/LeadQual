"""HTTP transport unit tests for _HttpTavilyProvider using httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from leadqual.evidence.search_client import (
    SearchExecutionError,
    SearchResult,
    TavilySearchClient,
    _HttpTavilyProvider,
)


def _make_provider_with_transport(
    handler: Any,
    *,
    api_key: str = "tvly-test-key",
    timeout: float = 5.0,
) -> tuple[_HttpTavilyProvider, list[httpx.Request]]:
    """Build an _HttpTavilyProvider backed by MockTransport and capture sent requests."""
    captured: list[httpx.Request] = []

    def wrapped_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped_handler)
    http_client = httpx.Client(transport=transport, timeout=timeout)
    provider = _HttpTavilyProvider(api_key=api_key, timeout=timeout, http_client=http_client)
    return provider, captured


def test_http_provider_sends_expected_payload_and_parses_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.tavily.com/search"
        body = json.loads(request.content.decode("utf-8"))
        assert body == {
            "api_key": "tvly-secret-123",
            "query": "Acme enterprise pricing",
            "max_results": 5,
            "search_depth": "basic",
            "include_raw_content": False,
        }
        return httpx.Response(
            status_code=200,
            json={
                "results": [
                    {
                        "url": "https://acme.example.com/pricing",
                        "title": "Acme Pricing",
                        "content": "Enterprise tier includes SAML SSO and audit logs.",
                        "score": 0.94,
                    }
                ]
            },
        )

    provider, captured = _make_provider_with_transport(handler, api_key="tvly-secret-123")
    raw = provider.search("Acme enterprise pricing", max_results=5)

    assert len(captured) == 1
    assert raw["results"][0]["url"] == "https://acme.example.com/pricing"


def test_search_client_with_mock_transport_end_to_end() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "results": [
                    {
                        "url": "https://acme.example.com/about",
                        "title": "About Acme",
                        "content": "B2B automation platform founded in 2021.",
                        "score": 0.88,
                    }
                ]
            },
        )

    provider, _ = _make_provider_with_transport(handler)
    client = TavilySearchClient(api_key="tvly-test-key", client=provider)

    results = client.search("About Acme")

    assert results == [
        SearchResult(
            url="https://acme.example.com/about",
            title="About Acme",
            snippet="B2B automation platform founded in 2021.",
            score=0.88,
        )
    ]


@pytest.mark.parametrize("status_code", [400, 401, 403, 429, 500, 503])
def test_search_client_wraps_http_error_statuses_in_search_execution_error(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            json={"error": f"Simulated HTTP {status_code}"},
        )

    provider, _ = _make_provider_with_transport(handler)
    client = TavilySearchClient(api_key="tvly-test-key", client=provider)

    with pytest.raises(SearchExecutionError) as exc_info:
        client.search("Acme funding")

    assert "Acme funding" in str(exc_info.value)


def test_search_client_wraps_transport_timeout_in_search_execution_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out reading from Tavily", request=request)

    provider, _ = _make_provider_with_transport(handler)
    client = TavilySearchClient(api_key="tvly-test-key", client=provider)

    with pytest.raises(SearchExecutionError):
        client.search("Acme SOC2")


def test_search_client_wraps_invalid_json_in_search_execution_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"<html><body>Bad Gateway</body></html>",
            headers={"content-type": "text/html"},
        )

    provider, _ = _make_provider_with_transport(handler)
    client = TavilySearchClient(api_key="tvly-test-key", client=provider)

    with pytest.raises(SearchExecutionError):
        client.search("Acme security")