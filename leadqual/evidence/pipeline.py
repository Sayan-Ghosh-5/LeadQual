"""End-to-end evidence collection pipeline: search -> SSRF-safe fetch/extract -> deduplicated store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from leadqual.evidence.fetcher import FetcherError, SafePageFetcher
from leadqual.evidence.search_client import SearchResult, TavilySearchClient
from leadqual.storage.evidence_repo import StoreEvidenceOutcome, store_fetched_document


@dataclass(frozen=True)
class UrlFetchFailure:
    """Record of a search candidate URL that failed fetching, SSRF validation, or extraction."""

    url: str
    error_type: str
    message: str


@dataclass
class ResearchRunReport:
    """Summary of a single search-and-collect execution for a lead."""

    lead_id: uuid.UUID
    query: str
    search_results: list[SearchResult] = field(default_factory=list)
    stored: list[StoreEvidenceOutcome] = field(default_factory=list)
    failures: list[UrlFetchFailure] = field(default_factory=list)

    @property
    def inserted_count(self) -> int:
        return sum(1 for item in self.stored if item.inserted)

    @property
    def deduplicated_count(self) -> int:
        return sum(1 for item in self.stored if not item.inserted)


@dataclass
class EvidenceResearchPipeline:
    """Coordinates search discovery, safe page fetching, and deduplicated PostgreSQL storage."""

    search_client: TavilySearchClient
    page_fetcher: SafePageFetcher

    def collect_for_lead(
        self,
        session: Session,
        *,
        lead_id: uuid.UUID,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> ResearchRunReport:
        """Search for candidate URLs, safely fetch/extract each, and persist deduplicated evidence."""
        search_results = self.search_client.search(
            query,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=False,
        )

        report = ResearchRunReport(
            lead_id=lead_id,
            query=query,
            search_results=search_results,
        )

        # Deduplicate candidate URLs within the same search batch before fetching
        seen_urls: set[str] = set()
        for hit in search_results:
            if hit.url in seen_urls:
                continue
            seen_urls.add(hit.url)

            try:
                document = self.page_fetcher.fetch_and_extract(hit.url)
            except FetcherError as exc:
                report.failures.append(
                    UrlFetchFailure(
                        url=hit.url,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            try:
                outcome = store_fetched_document(
                    session,
                    lead_id=lead_id,
                    document=document,
                )
            except ValueError as exc:
                report.failures.append(
                    UrlFetchFailure(
                        url=document.final_url,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            report.stored.append(outcome)

        return report