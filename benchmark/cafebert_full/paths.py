"""Shared paths for the reproducible CafeBERT benchmark package."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = Path(os.environ.get("S3_CAFEBERT_SOURCES_DIR", ROOT / "sources")).expanduser().resolve()
RESULTS = Path(os.environ.get("S3_CAFEBERT_RESULTS_DIR", ROOT / "results")).expanduser().resolve()
SMOKE_RESULTS = Path(os.environ.get("S3_CAFEBERT_SMOKE_RESULTS_DIR", ROOT / "smoke_results")).expanduser().resolve()
