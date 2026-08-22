"""Build auditable stage-wise timing tables and LaTeX snippets from CafeBERT runs.

The input CSV is never modified. Each reported cell first averages the five
topic counts within a (corpus, model, seed), then reports the mean ± sample SD
over the four registered seeds (11, 29, 42, 47).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    from paths import ROOT, RESULTS
INPUT = RESULTS / "full_results.csv"
AUDIT = RESULTS / "FULL_MULTISEED_AUDIT.md"
OUTPUT_DIR = RESULTS / "latex_timing"
CONFIG = ROOT / "cafebert_full_config.json"

CORPUS_ORDER = ["vietnamese-news", "visfd", "vi-medical", "vntc-it"]
CORPUS_LABELS = {
    "vietnamese-news": "Vietnamese-news",
    "visfd": "UIT-ViSFD",
    "vi-medical": "ViMedical Disease",
    "vntc-it": "VNTC--CNTT",
}
MODEL_ORDER = ["s3_axial", "s3_angular", "s3_combined", "lda", "nmf", "bertopic_kmeans"]
MODEL_LABELS = {
    "s3_axial": "S$^3$ axial",
    "s3_angular": "S$^3$ angular",
    "s3_combined": "S$^3$ combined",
    "lda": "LDA",
    "nmf": "NMF",
    "bertopic_kmeans": "BERTopic + UMAP + KMeans",
}
SEEDS = [11, 29, 42, 47]
TOPIC_COUNTS = [10, 20, 30, 40, 50]
STAGE_COLUMNS = [
    "ingest_preprocess_seconds_shared",
    "encoder_model_load_seconds_shared",
    "representation_seconds_cold_reference",
    "fit_seconds",
    "pipeline_seconds",
    "total_cold_seconds",
    "metric_seconds",
]
RENAME = {
    "ingest_preprocess_seconds_shared": "ingest_preprocess_s",
    "encoder_model_load_seconds_shared": "encoder_load_s",
    "representation_seconds_cold_reference": "representation_cold_s",
    "fit_seconds": "fit_topic_s",
    "pipeline_seconds": "pipeline_cold_ref_s",
    "total_cold_seconds": "total_cold_s",
    "metric_seconds": "metrics_s",
}


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def portable_repository_path(path: Path) -> str:
    """Return a repository-relative path when the artifact lives in this clone."""
    try:
        return path.resolve().relative_to(ROOT.parents[1]).as_posix()
    except ValueError:
        return path.as_posix()


def pm(mean: float, sd: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def load_and_validate() -> pd.DataFrame:
    if not AUDIT.exists() or "Status: **PASS**" not in AUDIT.read_text(encoding="utf-8"):
        raise RuntimeError("FULL_MULTISEED_AUDIT.md must show PASS before timing export.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_csv(INPUT)
    expected_rows = len(CORPUS_ORDER) * len(MODEL_ORDER) * len(SEEDS) * len(TOPIC_COUNTS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(frame)}.")
    if sorted(frame.seed.unique().tolist()) != SEEDS:
        raise RuntimeError("Seed coverage does not match the locked configuration.")
    if sorted(frame.n_topics.unique().tolist()) != TOPIC_COUNTS:
        raise RuntimeError("Topic-count coverage does not match the locked configuration.")
    if set(frame.model.unique()) != set(MODEL_ORDER) or set(frame.corpus.unique()) != set(CORPUS_ORDER):
        raise RuntimeError("Corpus or model coverage is incomplete.")
    if not frame.status.eq("ok").all():
        raise RuntimeError("Timing export cannot include a non-ok run.")
    if not np.isfinite(frame[STAGE_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("A timing value is non-finite.")

    # The runner's contract explicitly records these additive stages per row.
    pipeline_delta = frame.pipeline_seconds - (frame.representation_seconds_cold_reference + frame.fit_seconds)
    total_delta = frame.total_cold_seconds - (
        frame.ingest_preprocess_seconds_shared
        + frame.encoder_model_load_seconds_shared
        + frame.pipeline_seconds
    )
    if not np.allclose(pipeline_delta, 0.0, rtol=0, atol=1e-8):
        raise RuntimeError(f"pipeline_seconds reconciliation failed; max delta={pipeline_delta.abs().max()}")
    if not np.allclose(total_delta, 0.0, rtol=0, atol=1e-8):
        raise RuntimeError(f"total_cold_seconds reconciliation failed; max delta={total_delta.abs().max()}")
    if config["runtime_contract"]["network_download_excluded"] is not True:
        raise RuntimeError("The timing contract no longer excludes network download.")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        frame.groupby(["corpus", "model", "seed", "n_documents", "representation_kind"], as_index=False)[STAGE_COLUMNS]
        .mean()
        .rename(columns=RENAME)
    )
    stage_names = list(RENAME.values())
    aggregation: dict[str, tuple[str, str]] = {"n_documents": ("n_documents", "first")}
    for stage in stage_names:
        aggregation[f"{stage}_mean"] = (stage, "mean")
        aggregation[f"{stage}_sd"] = (stage, "std")
    summary = per_seed.groupby(["corpus", "model", "representation_kind"], as_index=False).agg(**aggregation)
    summary["run_count"] = len(SEEDS) * len(TOPIC_COUNTS)
    summary["seed_count"] = len(SEEDS)
    summary["k_count"] = len(TOPIC_COUNTS)
    summary["model_label"] = summary.model.map(MODEL_LABELS)
    summary["corpus_label"] = summary.corpus.map(CORPUS_LABELS)
    summary["corpus"] = pd.Categorical(summary.corpus, CORPUS_ORDER, ordered=True)
    summary["model"] = pd.Categorical(summary.model, MODEL_ORDER, ordered=True)
    return summary.sort_values(["corpus", "model"]).reset_index(drop=True)


def methods_markdown() -> str:
    return """# Timing appendix: CafeBERT benchmark

## Đơn vị báo cáo

Mỗi ô timing trước hết là trung bình qua `k = 10, 20, 30, 40, 50` trong cùng một tổ hợp **corpus × mô hình × seed**. Sau đó bảng báo `mean ± sample SD` qua bốn seed đã đăng ký trước: 11, 29, 42 và 47. Vì vậy, `n=4` cho độ lệch chuẩn; mỗi ô tóm tắt 20 phép chạy thô. Các kết quả chỉ áp dụng cho môi trường được ghi trong `environment.json`, corpus manifest và cấu hình hash `4bbba1f8131d9c8ed741219255d2985be219ecc9b9368ad84e025bcac1cd840b`.

| Stage ghi trong CSV | Nghĩa vận hành | Có thể diễn giải |
|---|---|---|
| `ingest_preprocess_seconds_shared` | Đọc corpus và tiền xử lý đã khóa, phân bổ theo corpus trong runner | Chi phí chuẩn bị văn bản, không phải thời gian fit topic |
| `encoder_model_load_seconds_shared` | Nạp checkpoint CafeBERT cục bộ cho hàng dùng CafeBERT; bằng 0 cho LDA/NMF | Chi phí khởi tạo encoder, không bao gồm tải mạng |
| `representation_seconds_cold_reference` | Mã hóa CafeBERT mean pooling cho S³/BERTopic, hoặc `CountVectorizer.fit_transform` cho LDA/NMF | Chi phí tạo biểu diễn lạnh theo lớp mô hình |
| `fit_seconds` | Fit mô hình và trích top-term sau khi representation sẵn sàng | So sánh chi phí phân tích/trích topic *warm cache* |
| `pipeline_seconds` | `representation_seconds_cold_reference + fit_seconds` | Chi phí pipeline sau khi corpus/model đã sẵn sàng, không gồm metric |
| `total_cold_seconds` | `ingest + model load + pipeline` | Cold-start có thể so sánh trong cấu hình hiện tại |
| `metric_seconds` | WEC-in, diversity, C_NPMI và audit alphabetic sau khi trích topic | Chi phí đánh giá, không được cộng vào pipeline hoặc fit |

## Quy tắc viết kết quả

Không gọi `fit_seconds` là “thời gian end-to-end”. Đây là thời gian fit và trích topic khi biểu diễn đã sẵn sàng. Khi dùng `pipeline_seconds`, phải ghi “cold-reference pipeline”, vì representation lạnh được đo một lần cho mỗi corpus/lớp biểu diễn và sao chép vào các hàng tương đương để không thay cache hit bằng 0 giây. `total_cold_seconds` thêm nạp dữ liệu, tiền xử lý và nạp model cục bộ. Network download bị loại khỏi toàn bộ phép đo.

LDA và NMF dùng `CountVectorizer`; S³ và BERTopic + UMAP + KMeans dùng CafeBERT. Vì vậy bảng tổng thời gian so sánh các pipeline mô hình hoàn chỉnh, không phải encoder ablation. Một claim hợp lệ có dạng: “Trên máy và cấu hình đã ghi, S³ combined có median/mean `fit_seconds` thấp hơn [phương pháp] trong [corpus]”. Không được suy ra rằng S³ luôn nhanh hơn mọi baseline hoặc nhanh hơn trên phần cứng khác.

## Tệp đầu ra

* `cafebert_stage_timing_summary.csv`: dữ liệu bảng đã tổng hợp, có mean/SD theo stage.
* `table_cafebert_timing.tex`: bảng LaTeX 24 hàng, đầy đủ corpus × model.
* `table_cafebert_timing_compact.tex`: bốn bảng LaTeX nhỏ theo corpus, dành cho phụ lục hoặc xoay ngang.
* `timing_validation.json`: các kiểm tra tái hợp thức stage, coverage và quy tắc gộp.

## Chèn vào LaTeX

Trong preamble, thêm `\\usepackage{booktabs}`, `\\usepackage{longtable}`, `\\usepackage{graphicx}` và `\\usepackage{pdflscape}`. Bảng 24 hàng trong `table_cafebert_timing.tex` đã được đặt trong `landscape`; bảng compact tự co theo `\\linewidth`. Không sửa số trực tiếp trong `.tex`; khi chạy lại benchmark hoặc thay CSV, chạy lại `python3 generate_cafebert_timing_appendix.py`.
"""


def full_latex_table(summary: pd.DataFrame) -> str:
    lines = [
        "% Generated by generate_cafebert_timing_appendix.py; do not edit values manually.",
        "\\begin{landscape}",
        "\\scriptsize",
        "\\begin{longtable}{llrrrr}",
        "\\caption{Stage-wise runtime (seconds, mean $\\pm$ sample SD over four seeds; each seed mean is over five $k$ values).}\\label{tab:cafebert-timing-stage}\\\\",
        "\\toprule",
        "Corpus & Method & Representation & Fit + topics & Pipeline cold-ref & Total cold \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Corpus & Method & Representation & Fit + topics & Pipeline cold-ref & Total cold \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row.corpus_label)),
                    str(row.model_label),
                    pm(row.representation_cold_s_mean, row.representation_cold_s_sd),
                    pm(row.fit_topic_s_mean, row.fit_topic_s_sd),
                    pm(row.pipeline_cold_ref_s_mean, row.pipeline_cold_ref_s_sd),
                    pm(row.total_cold_s_mean, row.total_cold_s_sd),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\multicolumn{6}{p{0.94\\linewidth}}{\\footnotesize Representation is CafeBERT mean encoding for S$^3$ and BERTopic + UMAP + KMeans, and CountVectorizer.fit\\_transform for LDA/NMF. Fit includes top-term extraction. Pipeline excludes metrics; total cold also includes corpus ingest/preprocessing and local model loading. Network download is excluded.}",
            "\\end{longtable}",
            "\\end{landscape}",
            "",
        ]
    )
    return "\n".join(lines)


def compact_latex_tables(summary: pd.DataFrame) -> str:
    blocks = ["% Generated by generate_cafebert_timing_appendix.py; one table per corpus."]
    for corpus in CORPUS_ORDER:
        part = summary.loc[summary.corpus.astype(str) == corpus]
        blocks.extend(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\resizebox{\\linewidth}{!}{%",
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "Method & Fit + topics & Pipeline cold-ref & Total cold " + "\\\\",
                "\\midrule",
            ]
        )
        for _, row in part.iterrows():
            blocks.append(
                " & ".join(
                    [
                        str(row.model_label),
                        pm(row.fit_topic_s_mean, row.fit_topic_s_sd),
                        pm(row.pipeline_cold_ref_s_mean, row.pipeline_cold_ref_s_sd),
                        pm(row.total_cold_s_mean, row.total_cold_s_sd),
                    ]
                )
                + " \\\\"
            )
        blocks.extend(
            [
                "\\bottomrule",
                "\\end{tabular}}",
                f"\\caption{{Stage-wise runtime on {latex_escape(CORPUS_LABELS[corpus])}. Seconds, mean $\\pm$ sample SD over four seeds; each seed is averaged over five $k$ values.}}",
                f"\\label{{tab:cafebert-timing-{corpus}}}",
                "\\end{table}",
                "",
            ]
        )
    return "\n".join(blocks)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_and_validate()
    summary = summarize(frame)
    summary.to_csv(OUTPUT_DIR / "cafebert_stage_timing_summary.csv", index=False)
    (OUTPUT_DIR / "table_cafebert_timing.tex").write_text(full_latex_table(summary), encoding="utf-8")
    (OUTPUT_DIR / "table_cafebert_timing_compact.tex").write_text(compact_latex_tables(summary), encoding="utf-8")
    (OUTPUT_DIR / "TIMING_METHODS_AND_LATEX.md").write_text(methods_markdown(), encoding="utf-8")
    validation = {
        "status": "pass",
        "source_csv": portable_repository_path(INPUT),
        "raw_rows": int(len(frame)),
        "summary_rows": int(len(summary)),
        "seed_count": len(SEEDS),
        "topic_count_values": TOPIC_COUNTS,
        "raw_runs_per_summary_cell": len(SEEDS) * len(TOPIC_COUNTS),
        "pipeline_identity": "representation_seconds_cold_reference + fit_seconds",
        "total_cold_identity": "ingest_preprocess_seconds_shared + encoder_model_load_seconds_shared + pipeline_seconds",
        "network_download_excluded": True,
    }
    (OUTPUT_DIR / "timing_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", **validation, "output_dir": str(OUTPUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
