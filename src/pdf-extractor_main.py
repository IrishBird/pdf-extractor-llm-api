"""Entry point for the PDF-to-Mongo loader.

Modes:
    --pdf PATH  Process one local PDF (dev / one-shot).
    (default)   Long-poll SQS for S3 Object Created events, download the PDF,
                run the pipeline, delete the message on success. Driven by S3 -> SQS.
"""

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Load PDFs from --pdf or from SQS (S3-triggered) messages."""
    _prepare_path()
    args = _parser().parse_args(argv)
    _load_env(_env_path())
    from pipeline.pdf_extractor_file_mongo import FileToMongo
    from pipeline.pdf_extractor_file_mongo import RunServices

    services = RunServices()
    loader = FileToMongo(services)
    if args.pdf:
        loader.execute(Path(args.pdf))
        return 0
    return _run_sqs(loader, once=args.once)


def _run_sqs(loader: object, once: bool) -> int:
    """Poll SQS forever (or once). Filename comes from the S3 event in the message."""
    from services.source.sqs_s3_source import SqsS3Source
    from utils.log_utils import get_logger

    logger = get_logger("pdf-extractor")
    source = SqsS3Source()
    logger.info("listening on SQS for S3 PDF events", extra={"queue": source.queue_url})
    while True:
        messages = source.receive()
        if not messages:
            if once:
                logger.info("no SQS messages; exiting (--once)")
                return 0
            continue
        for message in messages:
            ok = True
            for ref in message.objects:
                path = source.download(ref)
                try:
                    loader.execute(path)
                    logger.info("load successful", extra={"path": path})
                except Exception:
                    ok = False
                    logger.exception(
                        "load failed; leaving SQS message for retry",
                        extra={
                            "message_id": message.message_id,
                            "bucket": ref.bucket,
                            "key": ref.key,
                        },
                    )
                    break
            # if ok:
            #     source.delete(message)
        if once:
            return 0


def _parser() -> argparse.ArgumentParser:
    """Accept a local PDF path, or run as an SQS worker when omitted."""
    parser = argparse.ArgumentParser(
        description=(
            "Load benefit PDFs into Mongo. "
            "Pass --pdf for a local file, or omit it to poll SQS (S3 event driven)."
        )
    )
    parser.add_argument(
        "--pdf",
        required=False,
        help="Local path to one PDF (skips SQS)",
    )
    parser.add_argument(
        "--json",
        required=False,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="In SQS mode, process at most one poll cycle then exit",
    )
    return parser


def _prepare_path() -> None:
    """Put src on the import path for hyphenated entry modules."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _env_path() -> Path:
    """Return the project .env, then a .env in the working directory."""
    project = Path(__file__).resolve().parents[1] / ".env"
    if project.is_file():
        return project
    return Path(".env")


def _load_env(path: Path) -> None:
    """Set missing environment values from a local file. Do not log them."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        _apply_env_line(line)


def _apply_env_line(line: str) -> None:
    """Apply one KEY=VALUE line when that key is unset or empty."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return
    key, value = stripped.split("=", 1)
    name = key.strip()
    current = os.environ.get(name, "")
    if current:
        return
    os.environ[name] = value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
