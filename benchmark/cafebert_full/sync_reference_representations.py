#!/usr/bin/env python3
"""Copy audited representation caches into the repository reference artifact.

The script rewrites only the repository's reference/full_results.csv paths to
repository-relative POSIX paths and writes a checksum manifest. It never
modifies a source benchmark directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REFERENCE = PACKAGE / "reference"
TARGET_CACHE = REFERENCE / "representation_cache"
TARGET_RESULTS = REFERENCE / "full_results.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_reference_path(name: str) -> str:
    return f"benchmark/cafebert_full/reference/representation_cache/{name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync audited representation caches into the Git reference artifact")
    parser.add_argument("--source-cache", type=Path, required=True, help="representation_cache directory from an audited run")
    args = parser.parse_args()
    source_cache = args.source_cache.expanduser().resolve()
    if not source_cache.is_dir():
        raise SystemExit(f"Source cache directory does not exist: {source_cache}")

    with TARGET_RESULTS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "representation_reference_path" not in fieldnames:
        raise SystemExit("reference/full_results.csv lacks representation_reference_path")
    if len(rows) != 480:
        raise SystemExit(f"Expected 480 audited result rows; found {len(rows)}")

    names = sorted({Path(row["representation_reference_path"]).name for row in rows})
    if len(names) != 8:
        raise SystemExit(f"Expected 8 shared representation artifacts; found {len(names)}: {names}")

    TARGET_CACHE.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    for name in names:
        source = source_cache / name
        destination = TARGET_CACHE / name
        if not source.is_file():
            raise SystemExit(f"Missing source representation artifact: {source}")
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise SystemExit(f"Checksum mismatch after copy: {name}")
        manifest_files.append(
            {
                "path": relative_reference_path(name),
                "bytes": destination.stat().st_size,
                "sha256": destination_hash,
            }
        )

    for row in rows:
        name = Path(row["representation_reference_path"]).name
        row["representation_reference_path"] = relative_reference_path(name)

    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8", dir=TARGET_RESULTS.parent) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary_results = Path(handle.name)
    temporary_results.replace(TARGET_RESULTS)

    manifest = {
        "artifact_type": "audited_cafebert_representation_cache",
        "result_rows": len(rows),
        "unique_representation_artifacts": len(manifest_files),
        "files": manifest_files,
        "path_convention": "Repository-root-relative POSIX paths stored in reference/full_results.csv",
    }
    (REFERENCE / "representation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "artifacts": len(manifest_files), "target": str(TARGET_CACHE)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
