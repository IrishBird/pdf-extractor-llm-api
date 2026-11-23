"""Retrieval and structural scoring for vector eval."""

from __future__ import annotations

from typing import Any


def row_matches_case(row: dict[str, Any], case: dict[str, Any]) -> bool:
    """True when a retrieved row satisfies the golden-case expectations."""
    sections = case.get("expected_sectionNames") or []
    if sections:
        section = str(row.get("sectionName") or "")
        if section not in sections:
            return False

    path_needles = case.get("expected_path_contains") or []
    path = str(row.get("path") or "")
    if path_needles and not any(needle in path for needle in path_needles):
        return False

    text_needles = case.get("reference_text_contains") or []
    text = str(row.get("text") or "").casefold()
    if text_needles and not any(needle.casefold() in text for needle in text_needles):
        return False

    # If only a question was provided with no expectations, treat as unscored miss.
    if not sections and not path_needles and not text_needles:
        return False
    return True


def score_retrieval(
    cases: list[dict[str, Any]],
    retrieved_by_case: list[list[tuple[dict[str, Any], float]]],
    k: int,
) -> dict[str, Any]:
    """Compute Hit@k, Recall@k, and MRR over golden cases."""
    if len(cases) != len(retrieved_by_case):
        raise ValueError("cases and retrieved lists must align")
    if not cases:
        return {
            "k": k,
            "case_count": 0,
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "rows": [],
        }

    hit_count = 0
    recall_total = 0.0
    mrr_total = 0.0
    rows: list[dict[str, Any]] = []

    for case, ranked in zip[tuple[dict[str, Any], list[tuple[dict[str, Any], float]]]](
        cases, retrieved_by_case
    ):
        matches = [(row, score) for row, score in ranked if row_matches_case(row, case)]
        rank = next(
            (
                index + 1
                for index, (row, _score) in enumerate[tuple[dict[str, Any], float]](
                    ranked
                )
                if row_matches_case(row, case)
            ),
            None,
        )
        hit = rank is not None
        hit_count += int(hit)
        # Treat each case as one relevance set: recall is 1 if any hit, else 0,
        # unless multiple expected fatss are listed - then fraction of fatss hit.
        expected_fatss = _expected_fats_count(case)
        if expected_fatss <= 1:
            recall_total += 1.0 if hit else 0.0
        else:
            recall_total += min(len(matches), expected_fatss) / expected_fatss
        mrr_total += (1.0 / rank) if rank else 0.0
        rows.append(
            {
                "question": case.get("question", ""),
                "hit": hit,
                "rank": rank,
                "retrieved": [
                    {
                        "score": round(score, 4),
                        "sectionName": row.get("sectionName"),
                        "path": row.get("path"),
                        "accountNumber": row.get("accountNumber"),
                        "productNumber": row.get("productNumber"),
                        "text_preview": _preview(row.get("text")),
                    }
                    for row, score in ranked
                ],
            }
        )

    count = len(cases)
    return {
        "k": k,
        "case_count": count,
        "hit_at_k": hit_count / count,
        "recall_at_k": recall_total / count,
        "mrr": mrr_total / count,
        "rows": rows,
    }


def _expected_fats_count(case: dict[str, Any]) -> int:
    """How many distinct expectations a case encodes."""
    sections = case.get("expected_sectionNames") or []
    paths = case.get("expected_path_contains") or []
    texts = case.get("reference_text_contains") or []
    return max(len(sections), len(paths), len(texts), 1)


def _preview(text: object, limit: int = 160) -> str:
    """Shorten stored text for console reports."""
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
