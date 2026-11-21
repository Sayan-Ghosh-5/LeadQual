"""Unit tests for SSRF-safe page fetching and trafilatura text extraction."""

from __future__ import annotations

import httpx
import pytest

from leadqual.evidence.fetcher import (
    ContentExtractionError,
    FetchedDocument,
    FetchExecutionError,
    SafePageFetcher,
    UnsafeUrlError,
    extract_main_text,
    validate_url_target,
)


def _client_with_handler(handler, *, max_bytes: int = 2 * 1024 * 1024, max_redirects: int = 5) -> SafePageFetcher:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, follow_redirects=False)
    return SafePageFetcher(
        http_client=http_client,
        max_bytes=max_bytes,
        max_redirects=max_redirects,
        allow_dns_lookup=False,
    )


def test_normal_public_page_fetches_and_extracts_text() -> None:
    sample_html = """
    <html>
      <head><title>Acme Platform Overview</title></head>
      <body>
        <nav>Home | Pricing | Login</nav>
        <article>
          <h1>Acme Workflow Engine</h1>
          <p>Acme automates B2B onboarding and SOC2 compliance evidence collection for enterprise teams.</p>
          <p>Customers reduce manual qualification time across sales and RevOps pipelines.</p>
        </article>
        <footer>Copyright Acme Inc.</footer>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://acme.example.com/overview"
        return httpx.Response(status_code=200, text=sample_html, headers={"content-type": "text/html"})

    fetcher = _client_with_handler(handler)
    doc = fetcher.fetch_and_extract("https://acme.example.com/overview")

    assert isinstance(doc, FetchedDocument)
    assert doc.url == "https://acme.example.com/overview"
    assert doc.final_url == "https://acme.example.com/overview"
    assert doc.status_code == 200
    assert "Acme Workflow Engine" in doc.text
    assert "SOC2 compliance evidence" in doc.text


def test_safe_redirect_to_another_public_url_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://acme.example.com/start":
            return httpx.Response(
                status_code=302,
                headers={"location": "https://acme.example.com/docs/security"},
            )
        assert str(request.url) == "https://acme.example.com/docs/security"
        return httpx.Response(
            status_code=200,
            text="<html><body><article><h1>Security</h1><p>End-to-end encryption and SSO support.</p></article></body></html>",
        )

    fetcher = _client_with_handler(handler)
    doc = fetcher.fetch_and_extract("https://acme.example.com/start")

    assert doc.final_url == "https://acme.example.com/docs/security"
    assert "End-to-end encryption" in doc.text


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8080/health",
        "http://localhost/metrics",
        "http://sub.localhost/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://172.16.0.10/db",
        "http://192.168.1.1/router",
        "http://[::1]/loopback",
        "http://[fc00::1]/private",
        "http://[fe80::1]/link-local",
        "file:///etc/passwd",
        "ftp://example.com/archive.zip",
    ],
)
def test_private_loopback_and_disallowed_scheme_urls_are_rejected(blocked_url: str) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(status_code=200, text="should not reach")

    fetcher = _client_with_handler(handler)

    with pytest.raises(UnsafeUrlError):
        fetcher.fetch_and_extract(blocked_url)

    assert called is False


def test_redirect_pointing_to_loopback_or_metadata_ip_is_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=302,
            headers={"location": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"},
        )

    fetcher = _client_with_handler(handler)

    with pytest.raises(UnsafeUrlError):
        fetcher.fetch_and_extract("https://acme.example.com/bounce")


def test_oversized_response_stream_is_rejected() -> None:
    chunks = [b"<html><body><article><p>", b"A" * 5000, b"</p></article></body></html>"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            stream=httpx.ByteStream(b"".join(chunks)),
            headers={"transfer-encoding": "chunked"},
        )

    fetcher = _client_with_handler(handler, max_bytes=1024)

    with pytest.raises(FetchExecutionError, match="max_bytes"):
        fetcher.fetch_and_extract("https://acme.example.com/huge-page")


def test_oversized_content_length_header_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="<html><body><p>Small body</p></body></html>",
            headers={"content-length": "99999999"},
        )

    fetcher = _client_with_handler(handler, max_bytes=4096)

    with pytest.raises(FetchExecutionError, match="Content-Length"):
        fetcher.fetch_and_extract("https://acme.example.com/declared-huge")


def test_empty_or_unextractable_html_raises_content_extraction_error() -> None:
    with pytest.raises(ContentExtractionError):
        extract_main_text("<html><body></body></html>", url="https://acme.example.com/empty")


def test_dns_resolution_blocking_private_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(host: str, port: int, *args, **kwargs):
        return [(2, 1, 6, "", ("127.0.0.1", port))]

    monkeypatch.setattr("leadqual.evidence.fetcher.socket.getaddrinfo", fake_getaddrinfo)

    with pytest.raises(UnsafeUrlError, match="disallowed IP"):
        validate_url_target("https://evil-rebind.example.com/internal", allow_dns_lookup=True)