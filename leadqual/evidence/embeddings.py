"""Batch embedding step for populating Evidence.embedding vectors in PostgreSQL."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from leadqual.storage.models import Evidence


class EmbeddingError(Exception):
    """Base exception for embedding generation or dimension validation errors."""


class EmbeddingDimensionError(EmbeddingError, ValueError):
    """Raised when a returned embedding vector does not match the expected dimension."""


class EmbeddingProvider(Protocol):
    """Protocol for embedding backends so tests and callers can swap implementations."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class OpenAICompatibleEmbeddingProvider:
    """Calls an OpenAI-compatible /v1/embeddings endpoint (e.g., local LLM server or API)."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 30.0
    http_client: httpx.Client | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        endpoint = f"{self.base_url.rstrip('/')}/embeddings"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "input": texts}

        if self.http_client is not None:
            response = self.http_client.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError("Embedding endpoint returned an unexpected or mismatched data list.")

        vectors: list[list[float]] = []
        for entry in items:
            emb = entry.get("embedding") if isinstance(entry, dict) else None
            if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
                raise EmbeddingError("Embedding entry missing valid numeric vector.")
            vectors.append([float(x) for x in emb])
        return vectors


@dataclass(frozen=True)
class EmbeddingBatchFailure:
    """Records a batch that failed during provider execution or vector validation."""

    evidence_ids: list[uuid.UUID]
    error_type: str
    message: str


@dataclass
class BatchEmbeddingReport:
    """Summary of a batch embedding run over unembedded Evidence rows."""

    batches_processed: int = 0
    embedded_count: int = 0
    failures: list[EmbeddingBatchFailure] = field(default_factory=list)


@dataclass
class EvidenceBatchEmbedder:
    """Queries Evidence rows with NULL embedding, generates vectors in batches, and updates rows."""

    provider: EmbeddingProvider
    batch_size: int = 32
    expected_dimension: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.expected_dimension is not None and self.expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive when provided.")

    def _validate_vectors(self, vectors: list[list[float]], expected_count: int) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} embedding vectors, received {len(vectors)}."
            )

        dim = self.expected_dimension
        validated: list[list[float]] = []
        for idx, vec in enumerate(vectors):
            if not vec:
                raise EmbeddingDimensionError(f"Embedding at index {idx} is empty.")
            if dim is None:
                dim = len(vec)
            elif len(vec) != dim:
                raise EmbeddingDimensionError(
                    f"Embedding dimension mismatch at index {idx}: expected {dim}, got {len(vec)}."
                )
            validated.append([float(v) for v in vec])
        return validated

    def embed_pending_evidence(
        self,
        session: Session,
        *,
        lead_id: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> BatchEmbeddingReport:
        """Select Evidence rows where embedding IS NULL, embed in batches, and update rows."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided.")

        stmt = (
            select(Evidence.id, Evidence.passage)
            .where(Evidence.embedding.is_(None))
            .order_by(Evidence.retrieved_at.asc(), Evidence.id.asc())
        )
        if lead_id is not None:
            stmt = stmt.where(Evidence.lead_id == lead_id)
        if limit is not None:
            stmt = stmt.limit(limit)

        pending_rows = session.execute(stmt).all()
        report = BatchEmbeddingReport()

        if not pending_rows:
            return report

        for start in range(0, len(pending_rows), self.batch_size):
            chunk = pending_rows[start : start + self.batch_size]
            ids = [row.id for row in chunk]
            passages = [row.passage for row in chunk]

            try:
                raw_vectors = self.provider.embed_texts(passages)
                vectors = self._validate_vectors(raw_vectors, expected_count=len(passages))
            except EmbeddingError as exc:
                report.failures.append(
                    EmbeddingBatchFailure(
                        evidence_ids=ids,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue
            except Exception as exc:
                report.failures.append(
                    EmbeddingBatchFailure(
                        evidence_ids=ids,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            for ev_id, vector in zip(ids, vectors):
                session.execute(
                    update(Evidence)
                    .where(Evidence.id == ev_id)
                    .values(embedding=vector)
                )

            report.batches_processed += 1
            report.embedded_count += len(vectors)

        return report