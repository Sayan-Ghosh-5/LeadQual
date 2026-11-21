"""SSRF-safe HTTP page fetcher and trafilatura main-text extractor."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from pydantic import BaseModel, ConfigDict, Field


class FetcherError(Exception):
    """Base exception for page fetching and content extraction."""


class UnsafeUrlError(FetcherError, ValueError):
    """Raised when a target URL or redirect targets a disallowed scheme or private/internal IP."""


class FetchExecutionError(FetcherError):
    """Raised when an HTTP request fails, times out, or exceeds size limits."""


class ContentExtractionError(FetcherError):
    """Raised when clean text could not be extracted from fetched HTML."""


class FetchedDocument(BaseModel):
    """Normalized output of an SSRF-safe fetch and text extraction."""

    model_config = ConfigDict(frozen=True)

    url: str
    final_url: str
    status_code: int
    text: str = Field(..., min_length=1)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(datetime.utc))


def is_disallowed_ip(addr: str) -> bool:
    """Return True if an IPv4 or IPv6 address is non-public / SSRF-sensitive."""
    ip = ipaddress.ip_address(addr)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_site_local
    )


def validate_url_target(url: str, *, allow_dns_lookup: bool = True) -> str:
    """Validate URL scheme, host, and resolved IP addresses against SSRF rules."""
    cleaned = url.strip()
    if not cleaned:
        raise UnsafeUrlError("URL must not be empty.")

    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError(f"Disallowed URL scheme '{parsed.scheme}'; only http and https are allowed.")

    host = parsed.hostname
    if not host:
        raise UnsafeUrlError("URL is missing a valid hostname.")

    lowered_host = host.lower().rstrip(".")
    if lowered_host in {"localhost"} or lowered_host.endswith(".localhost") or lowered_host.endswith(".local"):
        raise UnsafeUrlError(f"Disallowed local hostname '{host}'.")

    # If host is a literal IP, inspect it directly.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if allow_dns_lookup:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                addr_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                raise FetchExecutionError(f"DNS resolution failed for host '{host}': {exc}") from exc

            if not addr_info:
                raise FetchExecutionError(f"DNS resolution returned no addresses for host '{host}'.")

            for entry in addr_info:
                resolved_ip = entry[4][0]
                if is_disallowed_ip(resolved_ip):
                    raise UnsafeUrlError(
                        f"Host '{host}' resolved to disallowed IP '{resolved_ip}'."
                    )
    else:
        if is_disallowed_ip(host):
            raise UnsafeUrlError(f"Direct IP target '{host}' is disallowed.")

    return cleaned


def extract_main_text(html: str, *, url: str | None = None) -> str:
    """Extract clean readable text from HTML using trafilatura."""
    if not html or not html.strip():
        raise ContentExtractionError("Fetched HTML body is empty.")

    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not extracted or not extracted.strip():
        raise ContentExtractionError(f"trafilatura could not extract text for URL '{url or '<unknown>'}'.")

    return extracted.strip()


@dataclass
class SafePageFetcher:
    """Fetches web pages safely with SSRF checks, bounded redirects, and byte limits."""

    timeout: float = 15.0
    max_bytes: int = 2 * 1024 * 1024  # 2 MiB default
    max_redirects: int = 5
    user_agent: str = "LeadQualBot/1.0 (+https://example.local)"
    http_client: httpx.Client | None = None
    allow_dns_lookup: bool = True

    def _read_body_with_limit(self, response: httpx.Response, current_url: str) -> bytes:
        content_length_hdr = response.headers.get("content-length")
        if content_length_hdr is not None:
            try:
                declared_size = int(content_length_hdr)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > self.max_bytes:
                raise FetchExecutionError(
                    f"Response Content-Length ({declared_size} bytes) exceeds limit ({self.max_bytes} bytes) for '{current_url}'."
                )

        chunks: list[bytes] = []
        total_read = 0
        for chunk in response.iter_bytes(chunk_size=16_384):
            if not chunk:
                continue
            total_read += len(chunk)
            if total_read > self.max_bytes:
                raise FetchExecutionError(
                    f"Response body exceeded max_bytes ({self.max_bytes}) while reading '{current_url}'."
                )
            chunks.append(chunk)

        return b"".join(chunks)

    def fetch_and_extract(self, url: str) -> FetchedDocument:
        """Validate URL, follow safe redirects, download bounded HTML, and extract text."""
        current_url = validate_url_target(url, allow_dns_lookup=self.allow_dns_lookup)

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        )
        

        try:
            redirects_followed = 0
            while True:
                validate_url_target(current_url, allow_dns_lookup=self.allow_dns_lookup)
                try:
                    with client.stream("GET", current_url, follow_redirects=False) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise FetchExecutionError(
                                    f"Redirect status {response.status_code} missing Location header for '{current_url}'."
                                )
                            redirects_followed += 1
                            if redirects_followed > self.max_redirects:
                                raise FetchExecutionError(
                                    f"Exceeded maximum redirects ({self.max_redirects}) starting from '{url}'."
                                )
                            current_url = urljoin(current_url, location)
                            validate_url_target(current_url, allow_dns_lookup=self.allow_dns_lookup)
                            continue

                        response.raise_for_status()
                        raw_bytes = self._read_body_with_limit(response, current_url)
                        encoding = response.encoding or "utf-8"
                        html = raw_bytes.decode(encoding, errors="replace")
                        text = extract_main_text(html, url=current_url)
                        fetched_at = datetime.now(timezone.utc)
                        return FetchedDocument(
                            url=url,
                            final_url=current_url,
                            status_code=response.status_code,
                            text=text,
                            fetched_at=fetched_at,
                        )
                except (UnsafeUrlError, FetchExecutionError, ContentExtractionError):
                    raise
                except httpx.HTTPError as exc:
                    raise FetchExecutionError(f"HTTP error fetching '{current_url}': {exc}") from exc
        finally:
            if owns_client:
                client.close()