"""Read embedding rows from the loader's vector Mongo collection."""

from __future__ import annotations

import math
import os
from typing import Any

from services.target.vector_target import build_vector_mongo
from services.target.vector_target import vector_collection_name

_REQUIRED_FIELDS = (
    "shardKey",
    "pk",
    "chunk",
    "path",
    "text",
    "embedding",
    "model",
    "loadTs",
    "runId",
)


class VectorStore:
    """Thin reader over VECTOR_MONGO_* for inventory and retrieval eval."""

    def __init__(self) -> None:
        self.db_name = os.environ["VECTOR_MONGO_DB"]
        self.collection_name = vector_collection_name()
        self._mongo = build_vector_mongo()

    def close(self) -> None:
        closer = getattr(self._mongo, "close", None)
        if callable(closer):
            closer()

    def fetch(
        self,
        *,
        account_number: str | None = None,
        product_number: str | None = None,
        section_name: str | None = None,
        shard_key: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching vector rows (embedding included)."""
        query: dict[str, object] = {}
        if account_number:
            query["accountNumber"] = account_number
        if product_number:
            query["productNumber"] = product_number
        if section_name:
            query["sectionName"] = section_name
        if shard_key:
            query["shardKey"] = shard_key
        result = self._mongo.find(
            db_name=self.db_name,
            collection_name=self.collection_name,
            filter=query or None,
            limit=limit,
        )
        if result is None:
            return []
        if isinstance(result, dict):
            return [result]
        return list[dict[str, Any]](result)

    def inventory(
        self,
        *,
        account_number: str | None = None,
        product_number: str | None = None,
        shard_key: str | None = None,
    ) -> dict[str, Any]:
        """Summarize stored rows and flag structural problems."""
        rows = self.fetch(
            account_number=account_number,
            product_number=product_number,
            shard_key=shard_key,
        )
        issues: list[str] = []
        by_section: dict[str, int] = {}
        by_account: dict[str, int] = {}
        dims: set[int] = set[int]()
        models: set[str] = set[str]()

        for index, row in enumerate[dict[str, Any]](rows):
            missing = [name for name in _REQUIRED_FIELDS if name not in row]
            if missing:
                issues.append(f"row[{index}] missing fields: {','.join(missing)}")
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                issues.append(f"row[{index}] empty embedding")
            elif not all(isinstance(value, (int, float)) for value in embedding):
                issues.append(f"row[{index}] non-numeric embedding")
            else:
                dims.add(len(embedding))
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                issues.append(f"row[{index}] empty text")
            section = str(row.get("sectionName") or "unknown")
            by_section[section] = by_section.get(section, 0) + 1
            account = str(row.get("accountNumber") or "")
            if account:
                by_account[account] = by_account.get(account, 0) + 1
            model = row.get("model")
            if isinstance(model, str) and model:
                models.add(model)

        if len(dims) > 1:
            issues.append(f"mixed embedding dimensions: {sorted(dims)}")

        return {
            "db": self.db_name,
            "collection": self.collection_name,
            "row_count": len(rows),
            "section_counts": dict[str, int](sorted(by_section.items())),
            "account_counts": dict[str, int](sorted(by_account.items())),
            "embedding_dimensions": sorted(dims),
            "models": sorted(models),
            "issue_count": len(issues),
            "issues": issues[:50],
        }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for two equal-length vectors."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip[tuple[float, float]](left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def rank_by_embedding(
    query_vector: list[float],
    rows: list[dict[str, Any]],
    k: int,
) -> list[tuple[dict[str, Any], float]]:
    """Return the top-k rows by cosine similarity to the query vector."""
    scored: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        score = cosine_similarity(query_vector, [float(v) for v in embedding])
        scored.append((row, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(k, 0)]
