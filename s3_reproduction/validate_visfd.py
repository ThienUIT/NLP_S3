"""External validation for ViSFD: correlate S3's unsupervised axes against the
dataset's own human-annotated aspect labels (BATTERY, CAMERA, DESIGN,
FEATURES, GENERAL, OTHERS, PERFORMANCE, PRICE, SCREEN, SER&ACC, STORAGE).

The paper's own benchmark datasets have no ground-truth topic labels, so its
evaluation relies entirely on diversity/coherence. ViSFD happens to ship
aspect tags per review, which lets us check something the paper couldn't:
does an axis discovered purely from embedding variance actually line up with
a real, independently-defined semantic category?

Method: for each axis, split documents into "mentions aspect X" vs "doesn't",
and measure how different their axis scores are (point-biserial correlation
+ mean difference). A strong |r| means the axis reliably separates documents
that discuss that aspect from ones that don't -- in whichever direction sign
that axis happens to point.

Usage:
    python -m s3_reproduction.validate_visfd artifacts/turftopic/visfd/models/model_n20_2223_180826.joblib
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from .checkpoint import load_model

AUC_TIERS = [(0.75, "tốt"), (0.65, "khá"), (0.0, "yếu")]


def auc_tier(auc: float) -> str:
    for threshold, label in AUC_TIERS:
        if auc >= threshold:
            return label
    return "yếu"

ASPECT_PATTERN = re.compile(r"\{([^}]+)\}")


def parse_aspects(label: object) -> set[str]:
    if not isinstance(label, str):
        return set()
    return {token.split("#")[0] for token in ASPECT_PATTERN.findall(label)}


def build_aspect_frame(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(csv_path)
    aspect_sets = frame["label"].map(parse_aspects)
    aspects = sorted({aspect for row in aspect_sets for aspect in row})
    for aspect in aspects:
        frame[aspect] = aspect_sets.map(lambda row, a=aspect: a in row)
    return frame, aspects


def correlate_axes(doc_topic: np.ndarray, frame: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    """r/p = association strength+significance (point-biserial correlation).
    auc = if you built a classifier "aspect mentioned?" purely by thresholding
    this axis's score (in whichever direction correlates positively), how well
    would it discriminate -- 0.5 is random, 1.0 is perfect. This is the number
    that actually answers "is this axis reliable enough to auto-tag with?",
    since r can look nontrivial while the classes still overlap heavily.
    """
    rows = []
    for axis in range(doc_topic.shape[1]):
        scores = doc_topic[:, axis]
        for aspect in aspects:
            mask = frame[aspect].to_numpy()
            if mask.sum() < 10 or (~mask).sum() < 10:
                continue
            r, p = stats.pointbiserialr(mask, scores)
            auc = roc_auc_score(mask, scores)
            auc = max(auc, 1 - auc)  # effective AUC in the axis's best-fit direction
            rows.append(
                {
                    "axis": axis,
                    "aspect": aspect,
                    "r": r,
                    "p": p,
                    "auc": auc,
                    "mean_diff": scores[mask].mean() - scores[~mask].mean(),
                    "n_mentioned": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def combined_axes_auc(doc_topic: np.ndarray, frame: pd.DataFrame, aspects: list[str], cv: int = 5) -> pd.DataFrame:
    """Instead of picking one 'best' axis per aspect, feed ALL axis scores into
    a small logistic regression and see how well the aspect can be predicted
    from the combination. An aspect's signal is often spread across several
    axes (see e.g. BATTERY splitting into 5 axes at n_topics=50) -- a single
    axis can look weak while the aspect is still very much recoverable from
    the full set. 5-fold cross-validated AUC, so this isn't just overfitting
    to the same data it's scored on.
    """
    rows = []
    for aspect in aspects:
        y = frame[aspect].to_numpy()
        if y.sum() < cv * 2 or (~y).sum() < cv * 2:
            continue
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, doc_topic, y, cv=cv, scoring="roc_auc")
        rows.append({"aspect": aspect, "combined_auc": scores.mean(), "combined_auc_std": scores.std()})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So khớp trục S3 với nhãn khía cạnh gốc của ViSFD, kèm AUC nếu dùng làm rule gán tag")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("dataset/ViSFD/ViSFD.csv"))
    parser.add_argument("--top-words", type=int, default=6)
    parser.add_argument("--export-csv", type=Path, default=None, help="Ghi toàn bộ bảng axis x aspect ra CSV")
    parser.add_argument(
        "--combined", action="store_true",
        help="Thêm cột AUC khi gộp TẤT CẢ trục vào 1 logistic regression (5-fold CV) thay vì chỉ chọn trục tốt nhất -- "
             "chậm hơn (train 11 model nhỏ) nhưng thường AUC cao hơn hẳn cho khía cạnh mà 1 trục đơn lẻ không nắm hết",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_model(args.checkpoint)
    dataset_dir = args.checkpoint.parent.parent
    document_embeddings = np.load(dataset_dir / "document_embeddings.npy")

    frame, aspects = build_aspect_frame(args.csv)
    n_docs = min(len(frame), len(document_embeddings))
    frame = frame.iloc[:n_docs].reset_index(drop=True)
    document_embeddings = document_embeddings[:n_docs]
    if len(frame) != len(document_embeddings):
        print(
            f"CẢNH BÁO: CSV có {len(frame)} dòng nhưng document_embeddings.npy có "
            f"{len(document_embeddings)} -- có thể lệch thứ tự, kết quả chỉ tham khảo."
        )

    doc_topic = checkpoint.ica.transform(document_embeddings)  # (n_docs, n_topics)
    results = correlate_axes(doc_topic, frame, aspects)

    print(f"n_topics={doc_topic.shape[1]}  n_docs={n_docs}  n_aspects={len(aspects)}\n")

    combined = combined_axes_auc(doc_topic, frame, aspects) if args.combined else None

    print("=== Khía cạnh gốc -> trục khớp nhất, xếp theo AUC (độ tin cậy nếu dùng làm rule gán tag) ===")
    summary_rows = []
    for aspect in aspects:
        rows = results[results["aspect"] == aspect]
        if rows.empty:
            print(f"{aspect}: không đủ mẫu để kiểm định")
            continue
        best = rows.loc[rows["auc"].idxmax()]
        words = checkpoint.topics[int(best["axis"])][: args.top_words] if best["mean_diff"] > 0 else checkpoint.topics_negative[int(best["axis"])][: args.top_words]
        pole = "cực dương" if best["mean_diff"] > 0 else "cực âm"
        tier = auc_tier(best["auc"])
        summary_rows.append((aspect, best, tier))
        combined_str = ""
        if combined is not None:
            match = combined[combined["aspect"] == aspect]
            if not match.empty:
                c = match.iloc[0]
                combined_str = f"  | gộp {doc_topic.shape[1]} trục: AUC={c['combined_auc']:.3f} ({auc_tier(c['combined_auc']):4s}, +/-{c['combined_auc_std']:.3f})"
        print(
            f"{aspect:10s} -> Topic {int(best['axis']):>2}  AUC={best['auc']:.3f} ({tier:4s})  "
            f"r={best['r']:+.3f}  {pole}  n={best['n_mentioned']:>4}  -- {words}{combined_str}"
        )

    n_reliable = sum(1 for _, _, tier in summary_rows if tier == "tốt")
    print(f"\n-> {n_reliable}/{len(summary_rows)} khía cạnh có AUC >= 0.75 khi chỉ dùng 1 trục tốt nhất")
    if combined is not None:
        n_reliable_combined = (combined["combined_auc"] >= 0.75).sum()
        print(f"-> {n_reliable_combined}/{len(combined)} khía cạnh có AUC >= 0.75 khi gộp tất cả trục (logistic regression)")

    if args.export_csv:
        results.sort_values(["aspect", "auc"], ascending=[True, False]).to_csv(args.export_csv, index=False)
        print(f"Đã ghi toàn bộ bảng axis x aspect vào {args.export_csv}")


if __name__ == "__main__":
    main()
