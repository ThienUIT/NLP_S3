#!/usr/bin/env python3
"""Build reproducible paper tables from the audited CafeBERT full results.

The input CSV is never modified.  The script expects the pre-registered full
matrix: 4 corpora x 6 models x 4 seeds x 5 topic counts = 480 rows.

Run from the repository root:
    python -m benchmark.cafebert_full.build_paper_tables

Outputs are written to benchmark/cafebert_full/reference/paper_tables/ by
default.  Use --input and --output-dir for a separately rerun benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "benchmark" / "cafebert_full" / "reference" / "full_results.csv"
DEFAULT_OUTPUT = ROOT / "benchmark" / "cafebert_full" / "reference" / "paper_tables"

MODEL_ORDER = [
    "s3_axial",
    "s3_angular",
    "s3_combined",
    "lda",
    "nmf",
    "bertopic_kmeans",
]
MODEL_LABELS = {
    "s3_axial": "S$^3$ axial",
    "s3_angular": "S$^3$ angular",
    "s3_combined": "S$^3$ combined",
    "lda": "LDA",
    "nmf": "NMF",
    "bertopic_kmeans": "BERTopic+UMAP+KMeans",
}
CORPUS_LABELS = {
    "vietnamese-news": "Vietnamese-news",
    "visfd": "UIT--ViSFD",
    "vi-medical": "ViMedical Disease",
    "vntc-it": "VNTC--CNTT",
}
EXPECTED_CORPORA = set(CORPUS_LABELS)
EXPECTED_SEEDS = {11, 29, 42, 47}
EXPECTED_K = {10, 20, 30, 40, 50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--primary-seed",
        type=int,
        default=42,
        help="Primary pre-specified seed for the main results table (default: 42).",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame, primary_seed: int) -> None:
    required = {
        "config_sha256",
        "corpus",
        "model",
        "seed",
        "n_topics",
        "status",
        "wec_in",
        "topic_diversity",
        "c_npmi",
        "fit_seconds",
        "pipeline_seconds",
        "total_cold_seconds",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 480:
        raise ValueError(f"Expected 480 benchmark rows, found {len(df)}")
    if set(df["status"].dropna()) != {"ok"}:
        raise ValueError("Only status=ok rows may enter paper tables")
    if df["config_sha256"].nunique() != 1:
        raise ValueError("Expected exactly one locked configuration hash")
    if set(df["corpus"]) != EXPECTED_CORPORA:
        raise ValueError(f"Unexpected corpus values: {sorted(set(df['corpus']))}")
    if set(df["model"]) != set(MODEL_ORDER):
        raise ValueError(f"Unexpected model values: {sorted(set(df['model']))}")
    if set(df["seed"]) != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected seed values: {sorted(set(df['seed']))}")
    if set(df["n_topics"]) != EXPECTED_K:
        raise ValueError(f"Unexpected topic counts: {sorted(set(df['n_topics']))}")
    if primary_seed not in EXPECTED_SEEDS:
        raise ValueError(f"Primary seed {primary_seed} is outside the locked seed set")

    key_counts = df.groupby(["corpus", "model", "seed", "n_topics"], dropna=False).size()
    if not (key_counts == 1).all():
        duplicates = key_counts[key_counts != 1]
        raise ValueError(f"Expected one row per benchmark key; violations: {duplicates.to_dict()}")

    numeric_columns = [
        "wec_in",
        "topic_diversity",
        "c_npmi",
        "fit_seconds",
        "pipeline_seconds",
        "total_cold_seconds",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.map(float).map(pd.notna).all():
            raise ValueError(f"Non-numeric value found in {column}")


def ordered_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = df.pivot(index=["corpus", "n_topics"], columns="model", values=metric)
    index = pd.MultiIndex.from_product(
        [list(CORPUS_LABELS), sorted(EXPECTED_K)], names=["corpus", "n_topics"]
    )
    return wide.reindex(index=index, columns=MODEL_ORDER)


def primary_table(df: pd.DataFrame, metric: str, primary_seed: int) -> pd.DataFrame:
    return ordered_wide(df.loc[df["seed"] == primary_seed], metric)


def mean_sd_table(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby(["corpus", "n_topics", "model"], as_index=False).agg(
        mean=(metric, "mean"),
        sd=(metric, "std"),
        n=(metric, "count"),
    )
    mean = ordered_wide(grouped.rename(columns={"mean": metric}), metric)
    sd = ordered_wide(grouped.rename(columns={"sd": metric}), metric)
    n = ordered_wide(grouped.rename(columns={"n": metric}), metric)
    if not (n == 4).all().all():
        raise ValueError("Sensitivity table requires all four seeds for every corpus/model/k")
    return mean, sd, n


def display_index(index: pd.MultiIndex) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(CORPUS_LABELS[corpus], f"{int(k)}") for corpus, k in index], names=["Corpus", "$k$"]
    )


def formatted_primary(wide: pd.DataFrame, decimals: int, bold_winner: bool = True) -> pd.DataFrame:
    formatted = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=object)
    for row_index, row in wide.iterrows():
        maximum = row.max(skipna=True)
        for model, value in row.items():
            rendered = f"{value:.{decimals}f}"
            if bold_winner and pd.notna(maximum) and value == maximum:
                rendered = rf"\textbf{{{rendered}}}"
            formatted.loc[row_index, model] = rendered
    formatted.index = display_index(formatted.index)
    formatted.columns = [MODEL_LABELS[model] for model in formatted.columns]
    return formatted


def formatted_mean_sd(mean: pd.DataFrame, sd: pd.DataFrame, decimals: int) -> pd.DataFrame:
    formatted = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=object)
    for row_index in mean.index:
        for model in mean.columns:
            formatted.loc[row_index, model] = (
                f"{mean.loc[row_index, model]:.{decimals}f} $\\pm$ {sd.loc[row_index, model]:.{decimals}f}"
            )
    formatted.index = display_index(formatted.index)
    formatted.columns = [MODEL_LABELS[model] for model in formatted.columns]
    return formatted


def write_latex(table: pd.DataFrame, output: Path, caption: str, label: str) -> None:
    body = table.to_latex(
        escape=False,
        multicolumn=True,
        multirow=True,
        column_format="ll" + "r" * len(table.columns),
    )
    text = "\n".join(
        [
            "% Generated by build_paper_tables.py. Do not hand-edit values.",
            "\\begin{table*}[t]",
            "\\centering",
            "\\scriptsize",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            body,
            "\\end{table*}",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")


def write_machine_csv(wide: pd.DataFrame, output: Path) -> None:
    exported = wide.reset_index()
    exported["corpus"] = exported["corpus"].map(CORPUS_LABELS)
    exported = exported.rename(columns={"n_topics": "k"})
    exported = exported.rename(columns=MODEL_LABELS)
    exported.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    validate_input(df, args.primary_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_hash = df["config_sha256"].iloc[0]
    primary_wec = primary_table(df, "wec_in", args.primary_seed)
    primary_diversity = primary_table(df, "topic_diversity", args.primary_seed)
    sensitivity_mean, sensitivity_sd, _ = mean_sd_table(df, "wec_in")
    cnmpi_mean, cnmpi_sd, _ = mean_sd_table(df, "c_npmi")
    timing_mean, timing_sd, _ = mean_sd_table(df, "fit_seconds")

    artifacts = [
        (
            "table_main_wec_in_seed42",
            primary_wec,
            formatted_primary(primary_wec, decimals=4),
            "WEC-in at the preregistered primary seed (42). Bold marks the row maximum only; it is not a statistical test.",
            "tab:cafebert-wec-in-seed42",
        ),
        (
            "table_companion_diversity_seed42",
            primary_diversity,
            formatted_primary(primary_diversity, decimals=3),
            "Topic diversity at the preregistered primary seed (42). Bold marks the row maximum only.",
            "tab:cafebert-diversity-seed42",
        ),
        (
            "table_sensitivity_wec_in_mean_sd",
            sensitivity_mean,
            formatted_mean_sd(sensitivity_mean, sensitivity_sd, decimals=4),
            "WEC-in mean $\\pm$ sample standard deviation across seeds 11, 29, 42, and 47.",
            "tab:cafebert-wec-in-sensitivity",
        ),
        (
            "table_appendix_cnmpi_mean_sd",
            cnmpi_mean,
            formatted_mean_sd(cnmpi_mean, cnmpi_sd, decimals=3),
            "C_NPMI mean $\\pm$ sample standard deviation across four seeds; reported as a robustness metric, not a model-selection criterion.",
            "tab:cafebert-cnmpi-appendix",
        ),
        (
            "table_timing_fit_seconds_mean_sd",
            timing_mean,
            formatted_mean_sd(timing_mean, timing_sd, decimals=2),
            "Fit-only time in seconds, mean $\\pm$ sample standard deviation across four seeds. Representation reuse is assumed; this is not end-to-end time.",
            "tab:cafebert-fit-time-sensitivity",
        ),
    ]

    for stem, raw, rendered, caption, label in artifacts:
        write_machine_csv(raw, args.output_dir / f"{stem}.csv")
        write_latex(rendered, args.output_dir / f"{stem}.tex", caption, label)

    readme = args.output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Paper tables generated from full_results.csv",
                "",
                f"- Input: `{args.input.as_posix()}`",
                f"- Rows: `{len(df)}`",
                f"- Configuration SHA-256: `{config_hash}`",
                f"- Primary seed: `{args.primary_seed}`",
                "- Sensitivity seeds: `11, 29, 42, 47`",
                "- Main metric: `wec_in`; C_NPMI is appendix/robustness only.",
                "- Timing table uses `fit_seconds`; cite pipeline/end-to-end timing separately.",
                "",
                "The script validates the complete 480-row matrix before writing any table.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote paper tables to {args.output_dir}")
    print(f"Validated {len(df)} rows with config hash {config_hash}")


if __name__ == "__main__":
    main()
