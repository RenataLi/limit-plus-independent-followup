"""Download the exact public files used by the experiment."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


FILES = {
    "limit_corpus.jsonl": (
        "https://huggingface.co/datasets/orionweller/LIMIT/resolve/"
        "215834026c13176e520b3bc9d0a055099537ef99/corpus.jsonl",
        "10209a6916e029c199676caec3349cb925e2a74451161899c30c36e1b9032f82",
    ),
    "limit_queries.jsonl": (
        "https://huggingface.co/datasets/orionweller/LIMIT/resolve/"
        "215834026c13176e520b3bc9d0a055099537ef99/queries.jsonl",
        "1ead4c54487728173aa1433778a2ab0f4cf1cf8aeeedb886c05238b32d818594",
    ),
    "limit_qrels.jsonl": (
        "https://huggingface.co/datasets/orionweller/LIMIT/resolve/"
        "215834026c13176e520b3bc9d0a055099537ef99/qrels.jsonl",
        "a4f9b25b694623c240c6499fb8d8a4896355db9c840f29bd14580d0d4100ea89",
    ),
    "limit_plus_queries.jsonl": (
        "https://raw.githubusercontent.com/informagi/Complex-Set-Compositional-IR/"
        "0a4105a328474d4a4c58b8e4fc613ec05c59fc22/code/data_generation_utils/"
        "limit_plus/limit_data/limit_quest_queries.jsonl",
        "c412df625e1530e81012e31e95fe6f339f5e7027f14acdffc8fe0e348eef55cd",
    ),
}

DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in FILES.items():
        destination = args.data_dir / filename
        if destination.exists() and sha256(destination) == expected:
            print(f"verified {filename}")
            continue
        fd, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=args.data_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "limitplus-audit/0.1"}
                    )
                    with urllib.request.urlopen(
                        request, timeout=DOWNLOAD_TIMEOUT_SECONDS
                    ) as response, temporary.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                    break
                except (OSError, TimeoutError, urllib.error.URLError):
                    temporary.unlink(missing_ok=True)
                    if attempt == DOWNLOAD_ATTEMPTS:
                        raise
                    time.sleep(attempt)
            actual = sha256(temporary)
            if actual != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
                )
            os.replace(temporary, destination)
            print(f"downloaded {filename}")
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
