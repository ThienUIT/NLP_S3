from __future__ import annotations

import argparse
import csv
import json
import math
import os
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


def plot_topic_table(dataset: str, result: dict, output: Path) -> None:
    apply_paper_style()
    topics = result["topics"]
    rows = [[f"Topic {index + 1}", ", ".join(words)] for index, words in enumerate(topics)]
    height = max(4.5, 0.43 * len(rows) + 1.6)
    figure, axis = plt.subplots(figsize=(14, height))
    axis.axis("off")
    table = axis.table(
        cellText=rows, colLabels=["Topic", "Top words"], colWidths=[0.12, 0.84],
        cellLoc="left", colLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_facecolor("#111827")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F4F6")
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
        "embedding_coherence", "aggregate_quality", "source",
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
                f"- Aggregate interpretability: {best['aggregate_quality']:.4f}", "",
            ]
        )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_results(args.input_dir)
    if not grouped:
        raise FileNotFoundError(f"No result JSON files found under {args.input_dir}")
    plot_metrics(grouped, args.output_dir)
    for dataset, rows in grouped.items():
        best = max(rows, key=lambda row: row["aggregate_quality"])
        plot_topic_table(dataset, best, args.output_dir)
    write_summary(grouped, args.output_dir)
    print(f"Visualizations written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
