#!/usr/bin/env python3
"""Materialize the pinned public CafeBERT checkpoint locally with a manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINT_ID = "uitnlp/CafeBERT"
REVISION = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned CafeBERT pretrained checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/cafebert_full/pretrained/CafeBERT"))
    args = parser.parse_args()
    destination = args.output_dir.expanduser().resolve()
    snapshot_download(repo_id=CHECKPOINT_ID, revision=REVISION, local_dir=destination)
    files = {path.relative_to(destination).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(destination.rglob("*")) if path.is_file()}
    manifest = {"checkpoint": CHECKPOINT_ID, "revision": REVISION, "files": files}
    (destination / "checkpoint_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint_dir": str(destination), "revision": REVISION, "files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
