#!/usr/bin/env python3
"""Generate reproducible analysis, charts and thesis-ready Markdown for the CafeBERT benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


try:
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    from paths import ROOT, RESULTS
CSV_PATH = RESULTS / "full_results.csv"
AUDIT_PATH = RESULTS / "FULL_MULTISEED_AUDIT.md"
CONFIG_PATH = ROOT / "cafebert_full_config.json"
REPORT_PATH = RESULTS / "S3_CAFEBERT_FULL_VIETNAMESE_REPORT.md"
SUMMARY_PATH = RESULTS / "cafebert_multiseed_summary.csv"
PRIMARY_PATH = RESULTS / "cafebert_seed42_primary.csv"
CHART_WEC_PATH = RESULTS / "cafebert-wec-in.png"
CHART_DIVERSITY_PATH = RESULTS / "cafebert-diversity.png"
CHART_TIMING_PATH = RESULTS / "cafebert-fit-timing.png"

TOPIC_COUNTS = [10, 20, 30, 40, 50]
SEEDS = [11, 29, 42, 47]
MODEL_ORDER = ["s3_axial", "s3_angular", "s3_combined", "lda", "nmf", "bertopic_kmeans"]
MODEL_LABELS = {
    "s3_axial": "S³ axial",
    "s3_angular": "S³ angular",
    "s3_combined": "S³ combined",
    "lda": "LDA",
    "nmf": "NMF",
    "bertopic_kmeans": "BERTopic + UMAP + KMeans",
}
COLORS = {
    "S³ axial": "#2046d8",
    "S³ angular": "#7c3aed",
    "S³ combined": "#65a30d",
    "LDA": "#b45309",
    "NMF": "#64748b",
    "BERTopic + UMAP + KMeans": "#dc2626",
}
CORPUS_ORDER = ["vietnamese-news", "visfd", "vi-medical", "vntc-it"]
CORPUS_LABELS = {
    "vietnamese-news": "Vietnamese-news",
    "visfd": "UIT-ViSFD",
    "vi-medical": "ViMedical Disease",
    "vntc-it": "VNTC-CNTT",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    return f"{fmt(mean, digits)} ± {fmt(0.0 if pd.isna(sd) else sd, digits)}"


def markdown_table(headers: list[str], rows: list[list[str]], numeric_columns: set[int] | None = None) -> str:
    numeric_columns = numeric_columns or set()
    alignment = ["---:" if index in numeric_columns else "---" for index in range(len(headers))]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(alignment) + "|",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def load_corpus_meta() -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}
    for corpus in CORPUS_ORDER:
        manifest = json.loads((RESULTS / f"{corpus}_manifest.json").read_text(encoding="utf-8"))
        source = manifest["source"]
        meta[corpus] = {
            "documents": int(manifest["n_documents"]),
            "id_hash": str(manifest["document_ids_sha256"]),
            "min_chars": int(manifest["minimum_text_chars"]),
            "repository": str(source["repository"]),
            "revision": str(source.get("snapshot_commit", source.get("revision", "N/A"))),
            "license": str(source.get("license", "Không công bố trong manifest")),
        }
    return meta


def prepare_frame() -> pd.DataFrame:
    frame = pd.read_csv(CSV_PATH)
    frame["method"] = frame["model"].map(MODEL_LABELS)
    frame["corpus_label"] = frame["corpus"].map(CORPUS_LABELS)
    return frame


def primary_metric_table(frame: pd.DataFrame, corpus: str, metric: str) -> str:
    pivot = (
        frame.loc[(frame["corpus"] == corpus) & (frame["seed"] == 42)]
        .pivot(index="n_topics", columns="method", values=metric)
        .reindex(index=TOPIC_COUNTS, columns=[MODEL_LABELS[model] for model in MODEL_ORDER])
    )
    rows: list[list[str]] = []
    for k, row in pivot.iterrows():
        winner = row.idxmax()
        displayed = [str(k)]
        for method, value in row.items():
            cell = fmt(value)
            displayed.append(f"**{cell}**" if method == winner else cell)
        rows.append(displayed)
    return markdown_table(
        ["k", *[MODEL_LABELS[model] for model in MODEL_ORDER]],
        rows,
        set(range(len(MODEL_ORDER) + 1)),
    )


def build_multiseed_summary(frame: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        frame.groupby(["corpus", "method", "seed"], as_index=False)
        .agg(
            wec_in=("wec_in", "mean"),
            topic_diversity=("topic_diversity", "mean"),
            c_npmi=("c_npmi", "mean"),
            fit_seconds=("fit_seconds", "mean"),
            pipeline_seconds=("pipeline_seconds", "mean"),
            total_cold_seconds=("total_cold_seconds", "mean"),
            metric_seconds=("metric_seconds", "mean"),
            alphabetic_term_rate=("alphabetic_term_rate", "mean"),
        )
    )
    summary = (
        per_seed.groupby(["corpus", "method"], as_index=False)
        .agg(
            wec_in_mean=("wec_in", "mean"),
            wec_in_sd=("wec_in", "std"),
            diversity_mean=("topic_diversity", "mean"),
            diversity_sd=("topic_diversity", "std"),
            c_npmi_mean=("c_npmi", "mean"),
            c_npmi_sd=("c_npmi", "std"),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_sd=("fit_seconds", "std"),
            pipeline_seconds_mean=("pipeline_seconds", "mean"),
            pipeline_seconds_sd=("pipeline_seconds", "std"),
            total_cold_seconds_mean=("total_cold_seconds", "mean"),
            metric_seconds_mean=("metric_seconds", "mean"),
            alpha_rate_mean=("alphabetic_term_rate", "mean"),
        )
    )
    return summary.assign(
        corpus=pd.Categorical(summary["corpus"], CORPUS_ORDER, ordered=True),
        method=pd.Categorical(summary["method"], [MODEL_LABELS[model] for model in MODEL_ORDER], ordered=True),
    ).sort_values(["corpus", "method"])


def sensitivity_table(summary: pd.DataFrame, corpus: str) -> str:
    rows: list[list[str]] = []
    part = summary.loc[summary["corpus"].astype(str) == corpus].set_index("method")
    for method in [MODEL_LABELS[model] for model in MODEL_ORDER]:
        row = part.loc[method]
        rows.append(
            [
                method,
                fmt_pm(row.wec_in_mean, row.wec_in_sd),
                fmt_pm(row.diversity_mean, row.diversity_sd),
                fmt_pm(row.c_npmi_mean, row.c_npmi_sd),
                fmt_pm(row.fit_seconds_mean, row.fit_seconds_sd, 2),
                fmt_pm(row.pipeline_seconds_mean, row.pipeline_seconds_sd, 2),
                fmt(row.alpha_rate_mean),
            ]
        )
    return markdown_table(
        ["Phương pháp", "WEC-in", "Diversity", "C_NPMI", "Fit-only (s)", "Pipeline cold-ref (s)", "Alphabetic rate"],
        rows,
        {1, 2, 3, 4, 5, 6},
    )


def timing_table(summary: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for corpus in CORPUS_ORDER:
        part = summary.loc[summary["corpus"].astype(str) == corpus].set_index("method")
        for method in [MODEL_LABELS[model] for model in MODEL_ORDER]:
            row = part.loc[method]
            rows.append(
                [
                    CORPUS_LABELS[corpus],
                    method,
                    fmt_pm(row.fit_seconds_mean, row.fit_seconds_sd, 2),
                    fmt_pm(row.pipeline_seconds_mean, row.pipeline_seconds_sd, 2),
                    fmt(row.total_cold_seconds_mean, 2),
                ]
            )
    return markdown_table(
        ["Corpus", "Phương pháp", "Fit-only warm (s)", "Pipeline cold-ref (s)", "Total cold (s)"],
        rows,
        {2, 3, 4},
    )


def timing_stage_definition_table() -> str:
    return markdown_table(
        ["Cột timing", "Nội dung đo", "Cách dùng đúng trong luận văn"],
        [
            ["`ingest_preprocess_seconds_shared`", "Đọc corpus và tiền xử lý đã khóa, ghi theo corpus", "Chi phí chuẩn bị dữ liệu; không phải fit topic"],
            ["`encoder_model_load_seconds_shared`", "Nạp CafeBERT cục bộ; bằng 0 cho LDA/NMF", "Chi phí khởi tạo encoder; không gồm tải mạng"],
            ["`representation_seconds_cold_reference`", "CafeBERT mean encoding cho S³/BERTopic hoặc CountVectorizer cho LDA/NMF", "Chi phí tạo biểu diễn lạnh theo lớp mô hình"],
            ["`fit_seconds`", "Fit mô hình và trích top-term sau khi biểu diễn sẵn sàng", "So sánh phân tích/trích topic với warm cache"],
            ["`pipeline_seconds`", "Representation cold-reference + fit", "Pipeline sau khi corpus/model đã sẵn sàng; không gồm metric"],
            ["`total_cold_seconds`", "Ingest + nạp model + pipeline", "Cold-start trong đúng môi trường đã ghi"],
            ["`metric_seconds`", "WEC-in, diversity, C_NPMI và alphabetic audit", "Chi phí đánh giá, không cộng vào fit/pipeline"],
        ],
    )


def count_wec_wins(frame: pd.DataFrame, seed_scope: list[int]) -> tuple[int, int, dict[str, int]]:
    scoped = frame.loc[frame["seed"].isin(seed_scope)]
    total = 0
    s3_wins = 0
    corpus_wins: dict[str, int] = {corpus: 0 for corpus in CORPUS_ORDER}
    for (corpus, _seed, _k), group in scoped.groupby(["corpus", "seed", "n_topics"], observed=True):
        max_score = group["wec_in"].max()
        winners = set(group.loc[group["wec_in"].eq(max_score), "model"])
        total += 1
        if any(model.startswith("s3_") for model in winners):
            s3_wins += 1
            corpus_wins[corpus] += 1
    return s3_wins, total, corpus_wins


def create_line_chart(frame: pd.DataFrame, metric: str, output: Path, ylabel: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), dpi=180, sharex=True)
    for ax, corpus in zip(axes.flat, CORPUS_ORDER, strict=True):
        part = frame.loc[frame["corpus"] == corpus]
        aggregate = part.groupby(["method", "n_topics"], as_index=False)[metric].agg(["mean", "std"]).reset_index()
        for model in MODEL_ORDER:
            method = MODEL_LABELS[model]
            values = aggregate.loc[aggregate["method"] == method].sort_values("n_topics")
            ax.plot(values["n_topics"], values["mean"], color=COLORS[method], marker="o", markersize=3.6, linewidth=1.9, label=method)
            sd = values["std"].fillna(0)
            ax.fill_between(values["n_topics"], values["mean"] - sd, values["mean"] + sd, color=COLORS[method], alpha=0.10)
        ax.set_title(CORPUS_LABELS[corpus], loc="left", fontweight="bold")
        ax.set_xticks(TOPIC_COUNTS)
        ax.set_xlabel("Số topic (k)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_fit_timing_chart(frame: pd.DataFrame, output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=180, sharey=False)
    for ax, corpus in zip(axes.flat, CORPUS_ORDER, strict=True):
        part = frame.loc[frame["corpus"] == corpus].groupby("method", as_index=False)["fit_seconds"].agg(["mean", "std"]).reset_index()
        part["method"] = pd.Categorical(part["method"], [MODEL_LABELS[model] for model in MODEL_ORDER], ordered=True)
        part = part.sort_values("method")
        bars = ax.bar(
            range(len(part)),
            part["mean"],
            yerr=part["std"].fillna(0),
            capsize=3,
            color=[COLORS[str(method)] for method in part["method"]],
            width=0.7,
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in part["mean"]], padding=3, fontsize=7, rotation=90)
        ax.set_title(CORPUS_LABELS[corpus], loc="left", fontweight="bold")
        ax.set_xticks(range(len(part)), ["S³ A", "S³ G", "S³ C", "LDA", "NMF", "B+U+K"], rotation=25, ha="right")
        ax.set_ylabel("Fit-only (s)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Thời gian fit-only trung bình qua seed × k (cache biểu diễn đã sẵn sàng)", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(frame: pd.DataFrame, summary: pd.DataFrame, meta: dict[str, dict[str, object]], config: dict[str, object]) -> None:
    primary = frame.loc[frame["seed"] == 42].copy()
    s3_primary_wins, primary_cells, _ = count_wec_wins(frame, [42])
    s3_all_wins, all_cells, all_corpus_wins = count_wec_wins(frame, SEEDS)
    provenance_rows = [
        [
            CORPUS_LABELS[corpus],
            f"{int(meta[corpus]['documents']):,}",
            f"≥ {meta[corpus]['min_chars']} ký tự",
            f"`{str(meta[corpus]['id_hash'])}`",
            f"[{str(meta[corpus]['repository']).replace('https://', '')}]({meta[corpus]['repository']})",
        ]
        for corpus in CORPUS_ORDER
    ]
    report = f"""# Báo cáo benchmark S³ với CafeBERT trên bốn corpus tiếng Việt

## Tóm tắt

Thí nghiệm đánh giá ba cấu hình **S³ — Semantic Signal Separation** (`axial`, `angular`, `combined`) cùng LDA, NMF và **BERTopic + UMAP + KMeans** trên bốn corpus tiếng Việt. Encoder cho S³ và BERTopic là checkpoint `uitnlp/CafeBERT`, pooling trung bình theo attention mask và chuẩn hóa L2. LDA/NMF dùng biểu diễn `CountVectorizer`, vì vậy đây là so sánh giữa các lớp mô hình topic, không phải ablation encoder.[1] [2]

Tập kết quả gồm **480 lần chạy thực**, tương ứng 4 corpus × 6 mô hình × 5 giá trị k × 4 seed. Kiểm toán cấu trúc xác nhận 480/480 hàng duy nhất, trạng thái `ok`, metric hữu hạn, số topic trả về đúng yêu cầu và mỗi topic có ít nhất 10 term. WEC-in là metric coherence chính theo protocol S³; topic diversity được báo song song và C_NPMI được giữ ở phụ lục như phép kiểm tra không đồng thuận.[1]

> **Kết quả ở phạm vi đã đo.** Ở seed 42, một biến thể S³ đạt WEC-in cao nhất trong **{s3_primary_wins}/{primary_cells}** ô *corpus × k*. Trên toàn bộ seed 11, 29, 42 và 47, S³ dẫn đầu WEC-in trong **{s3_all_wins}/{all_cells}** ô; phân rã theo corpus là Vietnamese-news {all_corpus_wins['vietnamese-news']}/20, UIT-ViSFD {all_corpus_wins['visfd']}/20, ViMedical {all_corpus_wins['vi-medical']}/20 và VNTC-CNTT {all_corpus_wins['vntc-it']}/20. Điều này là bằng chứng tái lập trên bốn corpus đã chọn, không phải chứng minh tính ưu việt phổ quát của S³.

## 1. Dữ liệu và provenance

{markdown_table(['Corpus', 'Tài liệu', 'Điều kiện lọc', 'SHA-256 document ID theo thứ tự', 'Nguồn snapshot'], provenance_rows, {1})}

Các nhãn sentiment trong UIT-ViSFD và nhãn trong Vietnamese-news không được đưa vào bước fit. Do đó, kết quả topic không được diễn giải như accuracy phân loại sentiment hoặc nhãn chủ đề có sẵn. Các manifest, hash document ID, revision nguồn và giấy phép đang lưu cùng kết quả để tái kiểm tra.

## 2. Protocol tái lập đã khóa

| Thành phần | Thiết lập |
|---|---|
| Mô hình S³ | `SemanticSignalSeparation` của Turftopic; `feature_importance = axial / angular / combined` [2] |
| Encoder ngữ nghĩa | `uitnlp/CafeBERT`; attention-mask mean pooling, L2-normalized, `max_length=256`, `batch_size=32`, xử lý batch theo độ dài tăng dần rồi trả về đúng thứ tự tài liệu [3] |
| Baseline lexical | LDA và NMF, `CountVectorizer(min_df=10)`, token pattern Unicode có hỗ trợ dấu gạch nối |
| Baseline embedding | BERTopic + UMAP (`n_neighbors=15`, `n_components=5`, `min_dist=0`, cosine) + KMeans (`n_init=10`), không phải BERTopic mặc định HDBSCAN [4] |
| Lưới | `k ∈ {{10, 20, 30, 40, 50}}`; 10 top-term/topic; seed chính 42; kiểm tra độ nhạy 11, 29, 42, 47 |
| WEC-in chính | Trung bình cosine của các cặp top-term trong từng topic, với Word2Vec huấn luyện trên chính corpus tiếng Việt |
| Chỉ số đi kèm | Topic diversity và alphabetic-term rate |
| Robustness phụ lục | Gensim C_NPMI. Metric này không dùng để chọn winner hoặc tạo chỉ số gộp |
| WEC-ex | **N/A**. Google News Word2Vec không phù hợp tiếng Việt |
| Điều kiện BERTopic | Vectorizer tạo topic của BERTopic dùng `min_df=2`, do `min_df=10` gây lỗi c-TF-IDF suy biến trong smoke test corpus 858 tài liệu. Thiết lập này không thay đổi tokenizer chung cho S³/LDA/NMF hay WEC-in/C_NPMI. |

`pipeline_seconds` là thời gian biểu diễn lạnh tham chiếu cộng fit và trích topic; `fit_seconds` là fit/trích topic khi biểu diễn đã sẵn sàng; `total_cold_seconds` cộng thêm ingest/preprocess và thời gian load model. Biểu diễn được cache theo corpus, nhưng thời gian lạnh gốc được sao chép vào mỗi hàng cùng loại mô hình để tránh đánh giá cache hit như 0 giây. Không cộng các cột này qua hàng để suy ra wall-clock tổng.

## 3. Kết quả chính: WEC-in, seed 42

Các ô đậm là WEC-in cao nhất trong cùng *corpus × k*. Đây là bảng seed chính; độ nhạy đa seed nằm ở mục 5.

### 3.1. Vietnamese-news

{primary_metric_table(primary, 'vietnamese-news', 'wec_in')}

### 3.2. UIT-ViSFD

{primary_metric_table(primary, 'visfd', 'wec_in')}

### 3.3. ViMedical Disease

{primary_metric_table(primary, 'vi-medical', 'wec_in')}

### 3.4. VNTC-CNTT

{primary_metric_table(primary, 'vntc-it', 'wec_in')}

![WEC-in trung bình ± SD qua bốn seed](cafebert-wec-in.png)

## 4. Topic diversity, seed 42

Topic diversity là tỉ lệ top-term khác nhau trên toàn bộ topic; giá trị cao không đồng nghĩa trực tiếp với coherence cao. Các ô đậm chỉ phương pháp cao nhất trong cùng *corpus × k*.

### 4.1. Vietnamese-news

{primary_metric_table(primary, 'vietnamese-news', 'topic_diversity')}

### 4.2. UIT-ViSFD

{primary_metric_table(primary, 'visfd', 'topic_diversity')}

### 4.3. ViMedical Disease

{primary_metric_table(primary, 'vi-medical', 'topic_diversity')}

### 4.4. VNTC-CNTT

{primary_metric_table(primary, 'vntc-it', 'topic_diversity')}

![Topic diversity trung bình ± SD qua bốn seed](cafebert-diversity.png)

## 5. Kiểm tra độ nhạy qua bốn seed

Mỗi seed trước hết được trung bình qua năm giá trị k; bảng sau là mean ± sample SD trên seed 11, 29, 42 và 47. WEC-in là cột diễn giải chính. C_NPMI được đưa vào nguyên trạng, kể cả khi thứ hạng không trùng WEC-in.

### 5.1. Vietnamese-news

{sensitivity_table(summary, 'vietnamese-news')}

### 5.2. UIT-ViSFD

{sensitivity_table(summary, 'visfd')}

### 5.3. ViMedical Disease

{sensitivity_table(summary, 'vi-medical')}

### 5.4. VNTC-CNTT

{sensitivity_table(summary, 'vntc-it')}

## 6. Timing theo giai đoạn

Bảng này ghi mean ± SD qua seed của các trung bình theo k. Cột **fit-only warm** phù hợp khi so sánh chi phí điều chỉnh mô hình sau khi biểu diễn đã có. Cột **pipeline cold-ref** và **total cold** thêm chi phí biểu diễn; vì các phương pháp ngữ nghĩa dùng CafeBERT còn LDA/NMF dùng CountVectorizer, các cột này phải được gắn đúng định nghĩa, không gọi chung là “speed” mà không nêu stage.

{timing_table(summary)}

{timing_stage_definition_table()}

Phụ lục `latex_timing/` được sinh trực tiếp từ `full_results.csv`: có bảng LaTeX 24 hàng theo *corpus × mô hình*, bảng compact theo từng corpus, CSV summary và kiểm tra hai đẳng thức `pipeline = representation + fit` cùng `total cold = ingest + model load + pipeline`. Mỗi ô trước hết trung bình qua năm giá trị k trong một seed, sau đó báo mean ± sample SD qua bốn seed. Không cộng các hàng này để suy ra wall-clock của toàn bộ benchmark.

![Fit-only trung bình ± SD](cafebert-fit-timing.png)

## 7. Diễn giải và giới hạn có thể dùng trong luận văn

Trong protocol này, kết quả 480 phép chạy hỗ trợ phát biểu hẹp: **S³ có thể vận hành với CafeBERT trên bốn corpus tiếng Việt và đạt WEC-in cao nhất ở {s3_all_wins}/{all_cells} ô corpus × seed × k đã đánh giá.** Đây là coherence dựa trên embedding nội bộ corpus, không phải đánh giá “đúng nhãn” của chủ đề. Vì vậy không nên suy diễn kết quả thành độ chính xác sentiment của UIT-ViSFD, chất lượng chẩn đoán y khoa của ViMedical hay hiệu quả tìm kiếm ngoài dữ liệu.

WEC-in, diversity và C_NPMI đo những thuộc tính khác nhau. C_NPMI có thể cho thứ hạng khác WEC-in do dựa vào đồng xuất hiện term; báo cáo giữ cả ba chỉ số thay vì chọn riêng metric có lợi. Không dùng `aggregate_proxy` ở bất kỳ bảng hay kết luận nào. WEC-ex vẫn N/A, bởi phép đo Google News Word2Vec không có cùng ngôn ngữ và corpus đích.

Kết quả thời gian chỉ so sánh được trong môi trường đã ghi tại `environment.json`, với đúng phiên bản thư viện, dữ liệu và cấu hình này. Khi muốn đưa claim thời gian vào luận văn, cần nêu rõ **fit-only warm** hoặc **pipeline cold-reference**, đồng thời nhấn mạnh LDA/NMF không dùng CafeBERT. Thí nghiệm không có gold-standard topic labels và chỉ giới hạn ở bốn corpus; đây là hai giới hạn cần giữ trong phần thảo luận.

## 8. Tái lập

Các artifact tham chiếu đã kiểm toán nằm trong `benchmark/cafebert_full/reference/`; một lần chạy mới ghi vào `benchmark/cafebert_full/results/` mặc định. Các file gồm `full_results.csv`, `full_topics.json`, `run_config.json`, corpus manifest, `environment.json`, `experiment_contract.json`, báo cáo audit, bảng summary và biểu đồ. Từ root của clone Git, lệnh kiểm toán và tái sinh báo cáo là:

```bash
export S3_CAFEBERT_RESULTS_DIR="$PWD/benchmark/cafebert_full/reference"
python -m benchmark.cafebert_full.audit_cafebert_full
python -m benchmark.cafebert_full.generate_cafebert_full_report
python -m benchmark.cafebert_full.generate_cafebert_timing_appendix
```

## Tài liệu tham khảo

[1]: https://aclanthology.org/2025.acl-long.32/ "Kardos et al. (2025), Semantic Signal Separation"
[2]: https://github.com/x-tabdeveloping/turftopic "Turftopic repository"
[3]: https://huggingface.co/uitnlp/CafeBERT "uitnlp/CafeBERT model card"
[4]: https://maartengr.github.io/BERTopic/ "BERTopic documentation"
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    if not AUDIT_PATH.exists() or "Status: **PASS**" not in AUDIT_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("FULL_MULTISEED_AUDIT.md must show PASS before report generation.")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = prepare_frame()
    expected_rows = len(CORPUS_ORDER) * len(MODEL_ORDER) * len(SEEDS) * len(TOPIC_COUNTS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(frame)}.")
    if not frame["status"].eq("ok").all():
        raise RuntimeError("Cannot report non-ok benchmark rows.")
    meta = load_corpus_meta()
    primary = frame.loc[frame["seed"] == 42].sort_values(["corpus", "model", "n_topics"])
    summary = build_multiseed_summary(frame)
    primary.to_csv(PRIMARY_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    create_line_chart(frame, "wec_in", CHART_WEC_PATH, "WEC-in")
    create_line_chart(frame, "topic_diversity", CHART_DIVERSITY_PATH, "Topic diversity")
    create_fit_timing_chart(frame, CHART_TIMING_PATH)
    write_report(frame, summary, meta, config)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "report": str(REPORT_PATH),
                "summary": str(SUMMARY_PATH),
                "charts": [str(CHART_WEC_PATH), str(CHART_DIVERSITY_PATH), str(CHART_TIMING_PATH)],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
