from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from pathlib import Path

# Keep Matplotlib's font cache inside the project on restricted Windows setups.
os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"visfd": "#2563EB", "vietnamese-news": "#E4572E"}
LABELS = {"visfd": "ViSFD", "vietnamese-news": "Vietnamese-News"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Turftopic S3 results")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/turftopic"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/visualizations"))
    return parser.parse_args()


def load_results(root: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(root.glob("*/topics_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("n_topics") not in {10, 20, 30, 40, 50}:
            continue
        data["source"] = str(path)
        data["aggregate_quality"] = math.sqrt(
            max(float(data["embedding_coherence"]), 0.0) * float(data["topic_diversity"])
        )
        timing = data.get("timing") or {}
        data["embedding_seconds"] = timing.get("embedding_seconds")
        data["model_seconds"] = timing.get("model_seconds")
        grouped.setdefault(str(data["dataset"]), []).append(data)

    # Ignore smoke tests/older runs: retain the largest document count per dataset.
    for dataset, rows in grouped.items():
        max_documents = max(int(row["documents"]) for row in rows)
        grouped[dataset] = sorted(
            (row for row in rows if int(row["documents"]) == max_documents),
            key=lambda row: int(row["n_topics"]),
        )
    return grouped


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_metrics(grouped: dict[str, list[dict]], output: Path) -> None:
    apply_paper_style()
    metrics = [
        ("embedding_coherence", "Embedding coherence"),
        ("topic_diversity", "Topic diversity"),
        ("aggregate_quality", "Aggregate interpretability"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        for dataset, rows in grouped.items():
            x = [int(row["n_topics"]) for row in rows]
            y = [float(row[metric]) for row in rows]
            axis.plot(
                x, y, marker="o", linewidth=2.4, markersize=6.5,
                color=COLORS.get(dataset), label=LABELS.get(dataset, dataset),
            )
            best = int(np.argmax(y))
            axis.scatter(x[best], y[best], s=125, facecolors="none",
                         edgecolors=COLORS.get(dataset), linewidths=2)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Number of topics")
        axis.set_xticks([10, 20, 30, 40, 50])
        axis.set_ylim(0.65, 1.01)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle("S³ with CafeBERT — Topic Quality", fontsize=16, fontweight="bold")
    figure.savefig(output / "s3_topic_quality.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "s3_topic_quality.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_timing(grouped: dict[str, list[dict]], output: Path) -> None:
    apply_paper_style()
    if not any(row.get("model_seconds") is not None for rows in grouped.values() for row in rows):
        return
    figure, axis = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    width = 0.35
    datasets = list(grouped.keys())
    for offset, (dataset, rows) in zip((-width / 2, width / 2), grouped.items()):
        x = np.arange(len(rows))
        model_seconds = [float(row.get("model_seconds") or 0) for row in rows]
        axis.bar(x + offset, model_seconds, width=width, color=COLORS.get(dataset), label=LABELS.get(dataset, dataset))
        embed_seconds = next((row.get("embedding_seconds") for row in rows if row.get("embedding_seconds")), None)
        if embed_seconds:
            axis.axhline(embed_seconds, color=COLORS.get(dataset), linestyle="--", linewidth=1.2, alpha=0.6)
    any_rows = next(iter(grouped.values()))
    axis.set_xticks(np.arange(len(any_rows)))
    axis.set_xticklabels([int(row["n_topics"]) for row in any_rows])
    axis.set_xlabel("Number of topics")
    axis.set_ylabel("Seconds")
    axis.set_title("Model fit time per n_topics (dashed = one-time embedding time)", fontweight="bold")
    axis.legend(frameon=False, loc="best")
    figure.savefig(output / "s3_timing.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "s3_timing.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_topic_table(dataset: str, result: dict, output: Path) -> None:
    apply_paper_style()
    topics = result["topics"]
    topics_negative = result.get("topics_negative") or [[] for _ in topics]
    has_negative = any(topics_negative)
    # Segmented Vietnamese words can be multi-word phrases ("bảo hành", "nhân
    # viên phục vụ"), so cells need wrapping now -- plain single-syllable
    # tokens used to fit on one line, compounds don't.
    wrap_width = 42 if has_negative else 90

    def wrap(words: list[str]) -> str:
        return textwrap.fill(", ".join(words), width=wrap_width) if words else ""

    if has_negative:
        rows = [
            [f"Topic {index + 1}", wrap(positive), wrap(negative)]
            for index, (positive, negative) in enumerate(zip(topics, topics_negative))
        ]
        col_labels = ["Topic", "Cực dương (+)", "Cực âm (-)"]
        col_widths = [0.06, 0.47, 0.47]
    else:
        rows = [[f"Topic {index + 1}", wrap(words)] for index, words in enumerate(topics)]
        col_labels = ["Topic", "Top words"]
        col_widths = [0.08, 0.88]
    line_counts = [max(cell.count("\n") + 1 for cell in row[1:]) for row in rows]
    line_height = 0.34
    row_heights = [max(1, n) * line_height + 0.18 for n in line_counts]
    height = max(4.5, sum(row_heights) + 1.8)
    figure, axis = plt.subplots(figsize=(20, height))
    axis.axis("off")
    table = axis.table(
        cellText=rows, colLabels=col_labels, colWidths=col_widths,
        cellLoc="left", colLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    n_cols = len(col_labels)
    total_height = sum(row_heights) + 0.5
    for row_index, row_height in enumerate(row_heights, start=1):
        for col in range(n_cols):
            table[row_index, col].set_height(row_height / total_height)
    for col in range(n_cols):
        table[0, col].set_height(0.5 / total_height)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.PAD = 0.01
        if row == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif has_negative and column == 2:
            cell.set_facecolor("#FDEDEB" if row % 2 else "#FBE0DC")
        elif has_negative and column == 1:
            cell.set_facecolor("#EAF3EA" if row % 2 else "#DCEBDC")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F4F6")
        cell.get_text().set_verticalalignment("center")
    label = LABELS.get(dataset, dataset)
    axis.set_title(
        f"{label}: {result['n_topics']} topics, {result['documents']:,} documents\n"
        f"Diversity={result['topic_diversity']:.3f}  |  "
        f"Coherence={result['embedding_coherence']:.3f}  |  "
        f"Aggregate={result['aggregate_quality']:.3f}",
        fontsize=14, fontweight="bold", pad=18,
    )
    figure.savefig(output / f"{dataset}_best_topics.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / f"{dataset}_best_topics.pdf", bbox_inches="tight")
    plt.close(figure)


def write_summary(grouped: dict[str, list[dict]], output: Path) -> None:
    fields = [
        "dataset", "documents", "n_topics", "topic_diversity",
        "embedding_coherence", "aggregate_quality",
        "embedding_seconds", "model_seconds", "source",
    ]
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rows in grouped.values():
            writer.writerows(rows)

    lines = ["# S³ CafeBERT result summary", ""]
    for dataset, rows in grouped.items():
        best = max(rows, key=lambda row: row["aggregate_quality"])
        lines.extend(
            [
                f"## {LABELS.get(dataset, dataset)}", "",
                f"- Documents: {best['documents']:,}",
                f"- Best topic count by geometric mean: {best['n_topics']}",
                f"- Diversity: {best['topic_diversity']:.4f}",
                f"- Coherence: {best['embedding_coherence']:.4f}",
                f"- Aggregate interpretability: {best['aggregate_quality']:.4f}",
            ]
        )
        if best.get("model_seconds") is not None:
            lines.append(
                f"- Timing (n_topics={best['n_topics']}): mã hoá {best['embedding_seconds']:.1f}s "
                f"+ fit model {best['model_seconds']:.2f}s"
            )
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_results(args.input_dir)
    if not grouped:
        raise FileNotFoundError(f"No result JSON files found under {args.input_dir}")
    plot_metrics(grouped, args.output_dir)
    plot_timing(grouped, args.output_dir)
    for dataset, rows in grouped.items():
        best = max(rows, key=lambda row: row["aggregate_quality"])
        plot_topic_table(dataset, best, args.output_dir)
    write_summary(grouped, args.output_dir)
    print(f"Visualizations written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
