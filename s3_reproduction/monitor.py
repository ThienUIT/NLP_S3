"""Anomaly monitoring for a stream of new (unlabeled) reviews against a
fitted S3 checkpoint -- the mechanism behind a "crisis alert" dashboard.

Not AUC: AUC needs ground-truth labels for the batch being scored, which
freshly-crawled comments don't have. What IS available without labels: how
far a new batch's axis scores sit from that axis' normal (training-time)
distribution. A sudden pile of comments landing far out on one axis --
compared to how rarely that happens on ordinary traffic -- is the anomaly
signal, independent of what the axis "means".

`validate_visfd.py`'s AUC/correlation table is still useful here, just at a
different point in the pipeline: computed ONCE, offline, against ViSFD's
real aspect labels, to attach a human-readable name ("BATTERY") to whichever
axis lights up, instead of just "Topic 23".
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoint import load_model
from .encoder import ENCODERS
from .validate_visfd import build_aspect_frame, correlate_axes


@dataclass
class AxisBaseline:
    mean: np.ndarray  # (n_topics,) average score per axis on the training corpus
    std: np.ndarray  # (n_topics,) spread per axis on the training corpus


def compute_baseline(document_topic: np.ndarray) -> AxisBaseline:
    """'Normal' per-axis behaviour, from the corpus the model was trained on."""
    return AxisBaseline(mean=document_topic.mean(axis=0), std=document_topic.std(axis=0))


def batch_zscores(batch_document_topic: np.ndarray, baseline: AxisBaseline) -> np.ndarray:
    """How many standard errors the batch's MEAN score (per axis) sits from
    the baseline mean -- a two-sample z-test against 'business as usual'.
    Large |z| means this batch talks about that axis unusually much (either
    pole), not explainable by ordinary sampling noise.
    """
    n = batch_document_topic.shape[0]
    standard_error = baseline.std / np.sqrt(max(n, 1))
    return (batch_document_topic.mean(axis=0) - baseline.mean) / np.clip(standard_error, 1e-9, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phát hiện bất thường trên 1 lô comment mới so với baseline lúc train")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("comments", type=Path, help="File .json chứa list[str] các comment mới")
    parser.add_argument("--csv", type=Path, default=Path("dataset/ViSFD/ViSFD.csv"))
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--min-auc-for-label", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_model(args.checkpoint)
    dataset_dir = args.checkpoint.parent.parent
    document_embeddings = np.load(dataset_dir / "document_embeddings.npy")
    document_topic = checkpoint.ica.transform(document_embeddings)
    baseline = compute_baseline(document_topic)

    # offline label calibration, reused (not recomputed against the new batch)
    frame, aspects = build_aspect_frame(args.csv)
    frame = frame.iloc[: len(document_topic)].reset_index(drop=True)
    results = correlate_axes(document_topic, frame, aspects)
    best_label: dict[int, tuple[str, float]] = {}
    for axis, rows in results.groupby("axis"):
        best = rows.loc[rows["auc"].idxmax()]
        if best["auc"] >= args.min_auc_for_label:
            best_label[int(axis)] = (best["aspect"], float(best["auc"]))

    encoder_kind = checkpoint.metadata.get("encoder_kind", "cafebert")
    encoder = ENCODERS[encoder_kind](model_name=checkpoint.metadata.get("encoder"), batch_size=8)
    comments = json.loads(args.comments.read_text(encoding="utf-8"))
    print(f"Lô mới: {len(comments)} comment. Baseline từ {len(document_topic)} review lúc train.\n")

    batch_embeddings = encoder.encode(comments, "batch")
    batch_topic = checkpoint.ica.transform(batch_embeddings)
    z = batch_zscores(batch_topic, baseline)

    # With N axes tested at once, some cross any fixed z-threshold by chance
    # alone (multiple-comparisons problem) -- an axis with no reliable label
    # is not actionable even if its z-score is large, since you don't know
    # what it means. Surface labeled axes first; unlabeled ones are shown
    # only as a count, not individually, to avoid a noisy false-alarm list.
    order = np.argsort(-np.abs(z))
    flagged = [a for a in order if abs(z[a]) >= args.z_threshold]
    labeled_flagged = [a for a in flagged if a in best_label]
    unlabeled_flagged = [a for a in flagged if a not in best_label]

    print("=== Cảnh báo diễn giải được (có nhãn từ hiệu chỉnh offline) ===")
    if labeled_flagged:
        print(f"{'Trục':>6}  {'z-score':>8}  {'Nhãn (AUC)':<22}  Từ khoá")
        for axis in labeled_flagged:
            pole = "+" if z[axis] > 0 else "-"
            aspect, auc = best_label[axis]
            words = checkpoint.topics[axis][:6] if z[axis] > 0 else checkpoint.topics_negative[axis][:6]
            print(f"{axis:>6}{pole}  {z[axis]:>8.2f}  {aspect} (AUC={auc:.2f}){'':<{max(0, 22 - len(aspect) - 12)}}  {words}")
    else:
        print("Không có trục nào có nhãn vượt ngưỡng.")

    print(
        f"\n({len(unlabeled_flagged)} trục khác cũng vượt ngưỡng nhưng chưa có nhãn tin cậy (AUC<{args.min_auc_for_label}) "
        "-- không hiện chi tiết vì không diễn giải được, dùng --min-auc-for-label thấp hơn nếu muốn xem)"
    )
    if not flagged:
        print("Không phát hiện bất thường vượt ngưỡng.")


if __name__ == "__main__":
    main()
