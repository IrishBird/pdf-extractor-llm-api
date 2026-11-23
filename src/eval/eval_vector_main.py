"""Evaluate vectors produced by pdf_extract_main.py.

Modes:
  inventory   Structural checks on VECTOR_MONGO_* rows
  retrieval   Hit@k / Recall@k / MRR against a golden question set
  all         Run inventory then retrieval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Run vector inventory and/or retrieval evaluation."""
    _prepare_path()
    args = _parser().parse_args(argv)
    _load_env(_env_path())

    from eval.metrics import score_retrieval
    from eval.vector_store import VectorStore
    from eval.vector_store import rank_by_embedding
    from services.middleware.embedding_api import embed_texts
    from utils.log_utils import get_logger

    logger = get_logger("pdf_extract")
    store = VectorStore()
    try:
        if args.mode in {"inventory", "all"}:
            report = store.inventory(
                account_number=args.account,
                product_number=args.product,
                shard_key=args.shard_key,
            )
            _print_inventory(report)
            if args.output:
                _write_json(args.output, {"inventory": report})
            if report["row_count"] == 0:
                logger.error("no vector rows found for the given filters")
                return 1
            if report["issue_count"] and args.fail_on_issues:
                return 2

        if args.mode in {"retrieval", "all"}:
            cases = _load_cases(args.golden)
            rows = store.fetch(
                account_number=args.account,
                product_number=args.product,
                shard_key=args.shard_key,
            )
            if not rows:
                logger.error("no vector rows available for retrieval eval")
                return 1
            retrieved: list[list[tuple[dict, float]]] = []
            for case in cases:
                question = str(case.get("question") or "").strip()
                if not question:
                    retrieved.append([])
                    continue
                scoped = _scope_rows(rows, case)
                query_vector = embed_texts([question])[0]
                ranked = rank_by_embedding(query_vector, scoped, args.k)
                retrieved.append(ranked)
            report = score_retrieval(cases, retrieved, args.k)
            _print_retrieval(report)
            if args.output:
                existing = {}
                path = Path(args.output)
                if path.is_file():
                    existing = json.loads(path.read_text(encoding="utf-8"))
                existing["retrieval"] = report
                _write_json(args.output, existing)
            if report["hit_at_k"] < args.min_hit_at_k:
                return 3
        return 0
    finally:
        store.close()


def _scope_rows(rows: list[dict], case: dict) -> list[dict]:
    """Optionally narrow candidate rows with case-level account/product filters."""
    account = case.get("accountNumber")
    product = case.get("productNumber")
    scoped = rows
    if account:
        scoped = [row for row in scoped if row.get("accountNumber") == account]
    if product:
        scoped = [row for row in scoped if row.get("productNumber") == product]
    return scoped or rows


def _load_cases(path: str | None) -> list[dict]:
    """Load golden questions from JSON."""
    file = Path(path) if path else Path(__file__).with_name("golden_dataset.json")
    raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"golden dataset must be a non-empty list: {file}")
    return raw


def _print_inventory(report: dict) -> None:
    print("=== Vector inventory ===")
    print(f"db/collection: {report['db']}/{report['collection']}")
    print(f"rows: {report['row_count']}")
    print(f"embedding dims: {report['embedding_dimensions']}")
    print(f"models: {report['models']}")
    print(f"sections: {report['section_counts']}")
    print(f"accounts: {report['account_counts']}")
    print(f"issues: {report['issue_count']}")
    for issue in report["issues"]:
        print(f"  - {issue}")


def _print_retrieval(report: dict) -> None:
    print("=== Retrieval eval ===")
    print(
        f"k={report['k']} cases={report['case_count']} "
        f"hit@{report['k']}={report['hit_at_k']:.2f} "
        f"recall@{report['k']}={report['recall_at_k']:.2f} "
        f"mrr={report['mrr']:.2f}"
    )
    for row in report["rows"]:
        mark = "HIT" if row["hit"] else "MISS"
        rank = row["rank"] if row["rank"] else "-"
        print(f"  [{mark}] rank={rank} {row['question']}")
        for item in row["retrieved"]:
            print(
                f"    {item['score']:.4f} "
                f"section={item['sectionName']} path={item['path']}"
            )
            print(f"      {item['text_preview']}")


def _write_json(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Mongo vector rows written by the PDF loader",
    )
    parser.add_argument(
        "--mode",
        choices=["inventory", "retrieval", "all"],
        default="all",
        help="Which evaluation to run (default: all)",
    )
    parser.add_argument(
        "--golden",
        help="Path to golden_dataset.json (default: src/eval/golden_dataset.json)",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    parser.add_argument("--account", help="Filter by accountNumber")
    parser.add_argument("--product", help="Filter by productNumber")
    parser.add_argument("--shard-key", help="Filter by PDF shardKey / filename")
    parser.add_argument(
        "--output",
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit non-zero when inventory finds structural issues",
    )
    parser.add_argument(
        "--min-hit-at-k",
        type=float,
        default=0.0,
        help="Exit non-zero when retrieval hit@k is below this threshold",
    )
    return parser


def _prepare_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _env_path() -> Path:
    project = Path(__file__).resolve().parents[2] / ".env"
    if project.is_file():
        return project
    return Path(".env")


def _load_env(path: Path) -> None:
    """Load .env into the process. Later keys in the file win; existing process env wins."""
    if not path.is_file():
        return
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip()] = value.strip()
    for name, value in parsed.items():
        if os.environ.get(name, ""):
            continue
        os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
