#!/usr/bin/env python3
"""Audit the complete CafeBERT benchmark matrix and its provenance."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


try:
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    from paths import ROOT, RESULTS
CSV_PATH = RESULTS / "full_results.csv"
CONFIG_PATH = ROOT / "cafebert_full_config.json"
REPORT_PATH = RESULTS / "FULL_MULTISEED_AUDIT.md"

EXPECTED_CORPORA = {"vietnamese-news", "visfd", "vi-medical", "vntc-it"}
EXPECTED_MODELS = {
    "s3_axial",
    "s3_angular",
    "s3_combined",
    "lda",
    "nmf",
    "bertopic_kmeans",
}
EXPECTED_SEEDS = {11, 29, 42, 47}
EXPECTED_TOPICS = {10, 20, 30, 40, 50}
EXPECTED_DOCUMENT_COUNTS = {
    "vietnamese-news": 858,
    "visfd": 10_000,
    "vi-medical": 12_060,
    "vntc-it": 3_571,
}
KEYS = ["corpus", "model", "seed", "n_topics", "config_sha256"]
METRICS = [
    "wec_in",
    "topic_diversity",
    "c_npmi",
    "fit_seconds",
    "pipeline_seconds",
    "total_cold_seconds",
    "metric_seconds",
]
REQUIRED_COLUMNS = set(
    KEYS
    + METRICS
    + [
        "status",
        "document_ids_sha256",
        "n_documents",
        "topic_count_returned",
        "min_topic_term_count",
    ]
)


def md_table(rows: list[tuple[object, ...]], headers: list[str]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|",
        *["| " + " | ".join(str(item) for item in row) + " |" for row in rows],
    ]


def main() -> None:
    frame = pd.read_csv(CSV_PATH)
    issues: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}.")
        REPORT_PATH.write_text(
            "# Audit full benchmark CafeBERT — all seeds\n\n"
            f"**FAIL**: missing required columns: {missing_columns}.\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "fail", "rows": len(frame), "issues": issues}, ensure_ascii=False))
        raise SystemExit(1)

    expected_rows = len(EXPECTED_CORPORA) * len(EXPECTED_MODELS) * len(EXPECTED_SEEDS) * len(EXPECTED_TOPICS)
    if len(frame) != expected_rows:
        issues.append(f"Expected {expected_rows} rows, found {len(frame)}.")

    duplicate_count = int(frame.duplicated(KEYS, keep=False).sum())
    if duplicate_count:
        issues.append(f"Found {duplicate_count} duplicate rows by benchmark key.")

    for label, observed, expected in [
        ("corpora", set(frame["corpus"].unique()), EXPECTED_CORPORA),
        ("models", set(frame["model"].unique()), EXPECTED_MODELS),
        ("seeds", set(frame["seed"].unique()), EXPECTED_SEEDS),
        ("topic grid", set(frame["n_topics"].unique()), EXPECTED_TOPICS),
    ]:
        if observed != expected:
            issues.append(f"Unexpected {label}: {sorted(observed)}.")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config_hashes = set(frame["config_sha256"].dropna().astype(str))
    if config_hashes != {expected_hash}:
        issues.append(
            "Config provenance mismatch: "
            f"expected {expected_hash}, observed {sorted(config_hashes)}."
        )

    non_ok = int((~frame["status"].eq("ok")).sum())
    if non_ok:
        issues.append(f"Non-ok rows: {non_ok}.")

    for field in METRICS:
        values = pd.to_numeric(frame[field], errors="coerce")
        invalid = int((~values.map(math.isfinite)).sum())
        if invalid:
            issues.append(f"{field} has {invalid} non-finite values.")

    if int((frame["fit_seconds"] <= 0).sum()):
        issues.append("At least one row has non-positive fit_seconds.")
    if int((frame["pipeline_seconds"] < frame["fit_seconds"]).sum()):
        issues.append("At least one row has pipeline_seconds below fit_seconds.")
    if int((frame["total_cold_seconds"] < frame["pipeline_seconds"]).sum()):
        issues.append("At least one row has total_cold_seconds below pipeline_seconds.")

    topic_mismatch = int((frame["topic_count_returned"] != frame["n_topics"]).sum())
    if topic_mismatch:
        issues.append(f"{topic_mismatch} rows returned an unexpected number of topics.")
    short_topics = int((frame["min_topic_term_count"] < 10).sum())
    if short_topics:
        issues.append(f"{short_topics} rows have a topic with fewer than 10 top terms.")

    coverage = frame.groupby(["corpus", "model", "seed"], observed=True).size()
    uneven_coverage = coverage.loc[coverage.ne(len(EXPECTED_TOPICS))]
    if not uneven_coverage.empty:
        issues.append(f"Cells without five k values: {uneven_coverage.to_dict()}.")

    actual_document_counts = frame.groupby("corpus", observed=True)["n_documents"].agg(lambda values: set(values))
    for corpus, expected_count in EXPECTED_DOCUMENT_COUNTS.items():
        observed_counts = actual_document_counts.get(corpus, set())
        if observed_counts != {expected_count}:
            issues.append(
                f"Unexpected document count for {corpus}: {sorted(observed_counts)}; "
                f"expected {expected_count}."
            )

    document_hashes = frame.groupby("corpus", observed=True)["document_ids_sha256"].agg(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    for corpus in EXPECTED_CORPORA:
        hashes = document_hashes.get(corpus, [])
        if len(hashes) != 1 or not hashes[0]:
            issues.append(f"Document-ID provenance is not unique for {corpus}: {hashes}.")

    per_corpus = frame.groupby("corpus", observed=True).size().sort_index()
    per_model = frame.groupby("model", observed=True).size().sort_index()
    per_seed = frame.groupby("seed", observed=True).size().sort_index()
    provenance_rows = [
        (corpus, EXPECTED_DOCUMENT_COUNTS[corpus], document_hashes.get(corpus, ["N/A"])[0])
        for corpus in sorted(EXPECTED_CORPORA)
    ]
    coverage_rows = [
        (corpus, model, seed, int(count))
        for (corpus, model, seed), count in coverage.sort_index().items()
    ]

    lines = [
        "# Audit full benchmark CafeBERT — all seeds",
        "",
        f"- Rows: **{len(frame)} / {expected_rows} expected**",
        f"- Unique benchmark keys: **{len(frame.drop_duplicates(KEYS))}**",
        f"- Seeds: **{', '.join(str(seed) for seed in sorted(EXPECTED_SEEDS))}**",
        f"- Config SHA-256: `{expected_hash}`",
        f"- Status: **{'PASS' if not issues else 'FAIL'}**",
        "",
        "## Coverage by corpus",
        "",
        *md_table([(corpus, int(count)) for corpus, count in per_corpus.items()], ["Corpus", "Rows"]),
        "",
        "## Coverage by model",
        "",
        *md_table([(model, int(count)) for model, count in per_model.items()], ["Model", "Rows"]),
        "",
        "## Coverage by seed",
        "",
        *md_table([(int(seed), int(count)) for seed, count in per_seed.items()], ["Seed", "Rows"]),
        "",
        "## Document-ID provenance",
        "",
        *md_table(provenance_rows, ["Corpus", "Documents", "SHA-256 of ordered document IDs"]),
        "",
        "## Cell coverage",
        "",
        "Every corpus × model × seed cell must contain exactly five topic counts (10, 20, 30, 40, 50).",
        "",
        *md_table(coverage_rows, ["Corpus", "Model", "Seed", "k values"]),
        "",
        "## Validation",
        "",
    ]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.extend(
            [
                "- All 480 expected configurations are unique, status `ok`, finite and structurally valid.",
                "- Each corpus × model × seed cell contains k = 10, 20, 30, 40, 50.",
                "- Each configuration returned its requested topic count with at least 10 terms per topic.",
                "- Document counts and ordered document-ID hashes are internally consistent for every corpus.",
                "- Timing invariants hold: pipeline time is not below fit-only time, and total cold time is not below pipeline time.",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass" if not issues else "fail", "rows": len(frame), "issues": issues}, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
