# CafeBERT full-corpus benchmark

This directory contains the runnable protocol behind the Vietnamese S³ benchmark. It does not replace `s3_reproduction/`, the existing demo, or the slide workflow. It adds a separate experiment package with six model configurations on four Vietnamese corpora:

| Group | Configurations |
|---|---|
| S³ | axial, angular, combined |
| Baselines | LDA, NMF, BERTopic + UMAP + KMeans |

Each configuration is evaluated at `k ∈ {10, 20, 30, 40, 50}`. Seed 42 is the primary slice; seeds 11, 29 and 47 are sensitivity runs. The complete grid has 480 runs.

## What the package records

`full_results.csv` stores one row per corpus/model/variant/seed/topic count and includes WEC-in, diversity, C_NPMI, document-ID hash, configuration hash, top terms, and timing stages. WEC-in is the primary coherence metric. C_NPMI is reported as a robustness metric and is not used to select a winner. WEC-ex is not applicable because the Google News vectors in the S³ paper are not Vietnamese.

Timing is deliberately split. `fit_seconds` is topic-model fitting after representation data is ready. `pipeline_seconds` is representation plus fit. `total_cold_seconds` also includes local data ingestion and initial encoder loading. Network download is outside all measured stages. A fit-only result must not be presented as end-to-end runtime.

## Setup

Use a separate virtual environment. The package relies on `unrar` to unpack the VNTC source archives.

```bash
python3 -m venv .venv-cafebert
source .venv-cafebert/bin/activate
python -m pip install --upgrade pip
python -m pip install -r benchmark/cafebert_full/requirements.txt
command -v unrar
```

Fetch the source snapshots and lock their revisions. This action is not included in timing.

```bash
python -m benchmark.cafebert_full.fetch_sources
python -m benchmark.cafebert_full.fetch_cafebert_checkpoint
export S3_CAFEBERT_CHECKPOINT_DIR="$PWD/benchmark/cafebert_full/pretrained/CafeBERT"
```

The source script writes `sources/sources.lock.json`; the checkpoint script writes `pretrained/CafeBERT/checkpoint_manifest.json`. Both source data and pretrained weights are ignored by Git.

## Run order

Start with the smoke grid. It checks loading, tokenization, UMAP/KMeans and metric finiteness before a long run.

```bash
python -m benchmark.cafebert_full.run_cafebert_smoke
python -m benchmark.cafebert_full.run_cafebert_full --seeds 42
python -m benchmark.cafebert_full.run_cafebert_full --seeds 11,29,47
```

The runner resumes by its configuration hash. Do not use `--force` unless a complete, deliberate re-run is intended. Outputs default to `benchmark/cafebert_full/results/`; set `S3_CAFEBERT_RESULTS_DIR` to move them onto a larger disk.

## Verify the committed reference artifact

The repository includes the audited 480-row reference result and eight shared representation files under `benchmark/cafebert_full/reference/`. This verification path does not require the raw corpora or CafeBERT weights. Set the result directory explicitly so the audit and report generators read the committed artifact rather than an empty fresh-run directory:

```bash
export S3_CAFEBERT_RESULTS_DIR="$PWD/benchmark/cafebert_full/reference"
python -m benchmark.cafebert_full.audit_cafebert_full
python -m benchmark.cafebert_full.generate_cafebert_full_report
python -m benchmark.cafebert_full.generate_cafebert_timing_appendix
```

Unset `S3_CAFEBERT_RESULTS_DIR` before starting a new experiment, or set it to a separate writable output directory.

## Audit and reports

```bash
python -m benchmark.cafebert_full.run_cafebert_full --dedupe-only
python -m benchmark.cafebert_full.audit_cafebert_full
python -m benchmark.cafebert_full.generate_cafebert_full_report
python -m benchmark.cafebert_full.generate_cafebert_timing_appendix
```

The audit checks expected coverage, duplicate resume keys, finite metrics, topic counts, term counts, consistent config hash and document-hash provenance. The report generator produces Markdown, CSV tables and plots. The timing generator produces `latex_timing/table_cafebert_timing.tex`, a compact table and a methods note for a thesis.

## Scope and wording

The benchmark is evidence for these four Vietnamese corpora only. It does not measure ground-truth topic accuracy and it does not establish that S³ is the fastest model overall. LDA and NMF do not use CafeBERT; their speed is a model-class comparison, not an encoder ablation. BERTopic in this package uses UMAP plus KMeans, not default HDBSCAN. Its vectorizer uses `min_df=2`; the lexical models use `min_df=10`.
