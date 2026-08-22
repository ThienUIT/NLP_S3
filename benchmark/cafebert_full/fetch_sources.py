#!/usr/bin/env python3
"""Fetch the pinned public sources used by the four-corpus benchmark.

Network acquisition is intentionally outside measured timing. The runner records
the source lock and document-ID hashes after local preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import requests

try:
    from .paths import SOURCES
except ImportError:  # pragma: no cover - direct invocation convenience
    from paths import SOURCES

VISFD_COMMIT = "4b11ec2e4e97839600e6035b49d1645c79023354"
SOURCES_TO_CLONE = {
    "ViMedical_Disease": ("https://github.com/PB3002/ViMedical_Disease.git", "2c2cb3909754a05346625d3b1aed609c1f5e0312"),
    "VNTC": ("https://github.com/duyvuleo/VNTC.git", "533a3d6e1a78d73cde5dcfaf867cf8fe62c1fca8"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_at_revision(destination: Path, url: str, revision: str) -> None:
    if not destination.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)


def download_visfd(destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    url = f"https://raw.githubusercontent.com/LuongPhan/UIT-ViSFD/{VISFD_COMMIT}/UIT-ViSFD.zip"
    temporary = destination.with_suffix(".partial")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def vietnamese_news_lock() -> dict[str, str]:
    response = requests.get("https://huggingface.co/api/datasets/vanhai123/vietnamese-news-dataset", timeout=45)
    response.raise_for_status()
    payload = response.json()
    return {
        "repository": "https://huggingface.co/datasets/vanhai123/vietnamese-news-dataset",
        "revision": str(payload.get("sha", "revision-not-returned")),
        "split": "train",
        "text_field": "content",
    }


def extract_vntc(repository: Path, destination: Path) -> list[dict[str, str]]:
    """Extract the pinned 27-topic VNTC archives to the loader's expected path."""
    archives = [
        repository / "Data" / "27Topics" / "Ver1.1" / "Train.rar",
        repository / "Data" / "27Topics" / "Ver1.1" / "Test.rar",
    ]
    if not all(archive.exists() for archive in archives):
        missing = [str(archive) for archive in archives if not archive.exists()]
        raise FileNotFoundError(f"Pinned VNTC archives were not found: {missing}")
    if not any(destination.rglob("*.txt")):
        destination.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            subprocess.run(["unrar", "x", "-o+", "-idq", str(archive), str(destination)], check=True)
    return [{"path": str(archive.relative_to(repository)), "sha256": sha256(archive)} for archive in archives]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pinned public sources for CafeBERT benchmark")
    parser.add_argument("--sources-dir", type=Path, default=SOURCES)
    args = parser.parse_args()
    root = args.sources_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in SOURCES_TO_CLONE.items():
        clone_at_revision(root / name, url, revision)
    archive = root / "UIT-ViSFD.zip"
    download_visfd(archive)
    vntc_archives = extract_vntc(root / "VNTC", root / "VNTC_extracted")
    lock = {
        "vietnamese_news": vietnamese_news_lock(),
        "uit_visfd": {"repository": "https://github.com/LuongPhan/UIT-ViSFD", "revision": VISFD_COMMIT, "archive": archive.name, "archive_sha256": sha256(archive)},
        "vi_medical": {"repository": "https://github.com/PB3002/ViMedical_Disease", "revision": SOURCES_TO_CLONE["ViMedical_Disease"][1]},
        "vntc": {
            "repository": "https://github.com/duyvuleo/VNTC",
            "revision": SOURCES_TO_CLONE["VNTC"][1],
            "archives": vntc_archives,
            "extracted_dir": "VNTC_extracted",
        },
    }
    (root / "sources.lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sources_dir": str(root), "lock": lock}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
