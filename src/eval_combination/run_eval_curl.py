"""Run retrieval eval using curl for embeddings (sandbox-friendly)."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _load_env() -> None:
    parsed: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return
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


def _embed_one(text: str) -> list[float]:
    payload = json.dumps(
        {
            "model": os.environ["LLM_API_EMBEDDING_MODEL"],
            "input": [text],
        }
    )
    proc = subprocess.run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            os.environ["LLM_API_EMBEDDING_ENDPOINT"],
            "-H",
            f"Authorization: Bearer {os.environ['LLM_API_EMBEDDING_KEY']}",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload,
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode() or f"curl exit {proc.returncode}")
    raw = proc.stdout
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    body = json.loads(raw.decode())
    return body["data"][0]["embedding"]


def _scope_rows(rows: list[dict], case: dict) -> list[dict]:
    account = case.get("accountNumber")
    product = case.get("productNumber")
    scoped = rows
    if account:
        scoped = [row for row in scoped if row.get("accountNumber") == account]
    if product:
        scoped = [row for row in scoped if row.get("productNumber") == product]
    return scoped or rows


def main() -> int:
    from eval.metrics import score_retrieval
    from eval.vector_store import VectorStore
    from eval.vector_store import rank_by_embedding

    _load_env()
    k = 3
    cases = json.loads((Path(__file__).with_name("golden_dataset.json")).read_text())
    print(
        f"cases={len(cases)} endpoint={os.environ.get('LLM_API_EMBEDDING_ENDPOINT')}",
        flush=True,
    )

    store = VectorStore()
    try:
        inventory = store.inventory(
            account_number="00001",
            product_number="FILE2002",
        )
        print(
            "inventory "
            f"rows={inventory['row_count']} "
            f"dims={inventory['embedding_dimensions']} "
            f"issues={inventory['issue_count']}",
            flush=True,
        )
        rows = store.fetch(account_number="00001", product_number="FILE2002")
    finally:
        store.close()

    if not rows:
        print("no vector rows", flush=True)
        return 1

    retrieved: list[list[tuple[dict, float]]] = []
    for index, case in enumerate[Any](cases, 1):
        question = str(case.get("question") or "").strip()
        if not question:
            retrieved.append([])
            print(f"[{index}/{len(cases)}] empty question", flush=True)
            continue
        vector = _embed_one(question)
        ranked = rank_by_embedding(vector, _scope_rows(rows, case), k)
        retrieved.append(ranked)
        print(f"[{index}/{len(cases)}] embedded ok", flush=True)

    report = score_retrieval(cases, retrieved, k)
    out = ROOT / "documents" / "output" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"inventory": inventory, "retrieval": report}, indent=2) + "\n"
    )

    print("=== Retrieval eval ===")
    print(
        f"k={report['k']} cases={report['case_count']} "
        f"answerable={report.get('answerable_case_count', report['case_count'])} "
        f"unanswerable={report.get('unanswerable_case_count', 0)} "
        f"hit@{report['k']}={report['hit_at_k']:.2f} "
        f"recall@{report['k']}={report['recall_at_k']:.2f} "
        f"mrr={report['mrr']:.2f}"
    )
    for name, stats in (report.get("by_category") or {}).items():
        print(
            f"  [{name}] n={stats['case_count']} "
            f"hit@{report['k']}={stats['hit_at_k']:.2f} "
            f"recall@{report['k']}={stats['recall_at_k']:.2f} "
            f"mrr={stats['mrr']:.2f}"
        )
    hits = sum(1 for row in report["rows"] if row.get("hit") is True)
    answerable = report.get("answerable_case_count", report["case_count"])
    print(f"hits={hits}/{answerable}")
    for row in report["rows"]:
        if row.get("answerable") is False:
            mark = "ABSTAIN"
            rank = "-"
        else:
            mark = "HIT" if row.get("hit") else "MISS"
            rank = row["rank"] if row.get("rank") else "-"
        category = row.get("category") or "-"
        print(f"  [{mark}] {category} rank={rank} {row['question'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
