"""Search client abstraction for retrieving web evidence via Tavily's REST API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from leadqual.config import settings

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchError(Exception):
    """Base exception for search-related failures."""


class InvalidSearchQueryError(SearchError, ValueError):
    """Raised when search parameters or API credentials fail upfront validation."""


class SearchExecutionError(SearchError):
    """Raised when the underlying search provider fails or returns malformed data."""


class SearchResult(BaseModel):
    """Normalized representation of a single web search hit."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., min_length=1)
    title: str = ""
    snippet: str = ""
    score: float | None = None


class SearchRequest(BaseModel):
    """Validated query parameters before invoking any search provider."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic")
    include_raw_content: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Search query must not be empty or whitespace-only.")
        return cleaned


class SearchProvider(Protocol):
    """Protocol for search backends so tests can supply fakes or fixtures."""

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
    ) -> dict[str, Any]: ...


@dataclass
class _HttpTavilyProvider:
    """Default HTTP adapter calling Tavily's /search JSON endpoint via httpx."""

    api_key: str
    timeout: float = 15.0
    http_client: httpx.Client | None = None

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_raw_content": include_raw_content,
        }

        if self.http_client is not None:
            response = self.http_client.post(TAVILY_SEARCH_URL, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(TAVILY_SEARCH_URL, json=payload)
                response.raise_for_status()
                data = response.json()

        if not isinstance(data, dict):
            raise SearchExecutionError("Tavily API returned a non-JSON-object response.")
        return data


@dataclass
class TavilySearchClient:
    """Wrapper around Tavily search enforcing input validation and error isolation."""

    api_key: str | None = None
    client: SearchProvider | None = None
    timeout: float = 15.0

    def __post_init__(self) -> None:
        if self.client is None:
            resolved_key = self.api_key if self.api_key is not None else settings.TAVILY_API_KEY
            if not resolved_key or not resolved_key.strip():
                raise InvalidSearchQueryError("TAVILY_API_KEY is missing or empty.")
            self.api_key = resolved_key.strip()
            self.client = _HttpTavilyProvider(api_key=self.api_key, timeout=self.timeout)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
    ) -> list[SearchResult]:
        """Validate inputs, run search via provider, and normalize results."""
        try:
            req = SearchRequest(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=include_raw_content,
            )
        except Exception as exc:
            raise InvalidSearchQueryError(str(exc)) from exc

        assert self.client is not None

        try:
            raw_response = self.client.search(
                req.query,
                max_results=req.max_results,
                search_depth=req.search_depth,
                include_raw_content=req.include_raw_content,
            )
        except SearchExecutionError:
            raise
        except Exception as exc:
            raise SearchExecutionError(f"Search provider failed for query '{req.query}': {exc}") from exc

        if not isinstance(raw_response, dict) or not isinstance(raw_response.get("results"), list):
            raise SearchExecutionError("Search provider returned an unexpected response payload.")

        results: list[SearchResult] = []
        for item in raw_response["results"]:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url or not isinstance(url, str):
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=str(item.get("title") or ""),
                    snippet=str(item.get("content") or item.get("snippet") or ""),
                    score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
                )
            )

        return results