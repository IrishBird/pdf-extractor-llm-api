"""Retrieval and structural scoring for vector eval."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def row_matches_case(row: dict[str, Any], case: dict[str, Any]) -> bool:
    """True when a retrieved row satisfies the golden-case expectations."""
    if not _is_answerable(case):
        return False

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
    """Compute Hit@k, Recall@k, and MRR over golden cases.

    Unanswerable cases ('answerable: false') are excluded from Hit@k / Recall@k / MRR
    and reported separately under 'unanswerable' for abstention-oriented eval.
    """
    if len(cases) != len(retrieved_by_case):
        raise ValueError("cases and retrieved lists must align")
    if not cases:
        return {
            "k": k,
            "case_count": 0,
            "answerable_case_count": 0,
            "unanswerable_case_count": 0,
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "by_category": {},
            "unanswerable": {"case_count": 0, "rows": []},
            "rows": [],
        }

    hit_count = 0
    recall_total = 0.0
    mrr_total = 0.0
    answerable_count = 0
    rows: list[dict[str, Any]] = []
    unanswerable_rows: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"case_count": 0, "hit_count": 0, "recall_total": 0.0, "mrr_total": 0.0}
    )

    for case, ranked in zip(cases, retrieved_by_case):
        category = str(case.get("category") or "unspecified")
        preview = {
            "question": case.get("question", ""),
            "category": category,
            "answerable": _is_answerable(case),
            "retrieved": [
                {
                    "score": round(score, 4),
                    "sectionName": row.get("sectionName"),
                    "path": row.get("path"),
                    "accountNumber": row.get("accountNumber"),
                    "productNumber": row.get("productNumber"),
                    "text_preview": _preview(row.get("text")),
                    "referenced_text": row.get("referenced_text"),
                }
                for row, score in ranked
            ],
        }

        if not _is_answerable(case):
            preview["hit"] = None
            preview["rank"] = None
            preview["expected_behavior"] = case.get("expected_behavior") or "abstain"
            unanswerable_rows.append(preview)
            rows.append(preview)
            continue

        matches = [(row, score) for row, score in ranked if row_matches_case(row, case)]
        rank = next(
            (
                index + 1
                for index, (row, _score) in enumerate(ranked)
                if row_matches_case(row, case)
            ),
            None,
        )
        hit = rank is not None
        answerable_count += 1
        hit_count += int(hit)
        # Treat each case as one relevance set: recall is 1 if any hit, else 0,
        # unless multiple expected facets are listed - then fraction of facets hit.
        expected_facets = _expected_facet_count(case)
        if expected_facets <= 1:
            recall = 1.0 if hit else 0.0
        else:
            recall = min(len(matches), expected_facets) / expected_facets
        recall_total += recall
        mrr = (1.0 / rank) if rank else 0.0
        mrr_total += mrr

        stats = category_stats[category]
        stats["case_count"] += 1
        stats["hit_count"] += int(hit)
        stats["recall_total"] += recall
        stats["mrr_total"] += mrr

        preview["hit"] = hit
        preview["rank"] = rank
        rows.append(preview)

    by_category = {
        name: {
            "case_count": int(stats["case_count"]),
            "hit_at_k": (
                (stats["hit_count"] / stats["case_count"])
                if stats["case_count"]
                else 0.0
            ),
            "recall_at_k": (
                (stats["recall_total"] / stats["case_count"])
                if stats["case_count"]
                else 0.0
            ),
            "mrr": (
                (stats["mrr_total"] / stats["case_count"])
                if stats["case_count"]
                else 0.0
            ),
        }
        for name, stats in sorted(category_stats.items())
    }

    return {
        "k": k,
        "case_count": len(cases),
        "answerable_case_count": answerable_count,
        "unanswerable_case_count": len(unanswerable_rows),
        "hit_at_k": (hit_count / answerable_count) if answerable_count else 0.0,
        "recall_at_k": (recall_total / answerable_count) if answerable_count else 0.0,
        "mrr": (mrr_total / answerable_count) if answerable_count else 0.0,
        "by_category": by_category,
        "unanswerable": {
            "case_count": len(unanswerable_rows),
            "rows": unanswerable_rows,
        },
        "rows": rows,
    }


def _is_answerable(case: dict[str, Any]) -> bool:
    """False when the golden case expects abstention / no grounded answer."""
    if case.get("answerable") is False:
        return False
    if str(case.get("category") or "") == "unanswerable":
        return False
    if str(case.get("expected_behavior") or "").casefold() == "abstain":
        return False
    return True


def _expected_facet_count(case: dict[str, Any]) -> int:
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
