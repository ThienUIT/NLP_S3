"""Streamlit demo: chiếu một đoạn văn bất kỳ lên các trục ngữ nghĩa mà S3 đã học.

Suy luận cho tài liệu mới dùng đúng công thức cuối Section 3.1 của paper:
    encode văn bản mới -> X_hat, rồi S_hat = X_hat . C^T  (== FastICA.transform)
không train lại gì -- chỉ tái sử dụng checkpoint đã lưu bởi s3_reproduction.cli.

Mỗi trục có hai cực (paper §3.1: "negative definition" của topic) -- checkpoint
lưu cả từ khoá cực dương và cực âm, demo hiển thị cả hai.

Chạy: streamlit run demo/app.py   (hoặc `make demo`)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s3_reproduction.checkpoint import ModelCheckpoint, load_model
from s3_reproduction.encoder import ENCODERS
from s3_reproduction.inspect import WordScore, format_word_scores, rank_vocabulary
from s3_reproduction.monitor import batch_zscores, compute_baseline
from s3_reproduction.validate_uts_bank import build_label_frame as build_bank_label_frame
from s3_reproduction.validate_visfd import build_aspect_frame, correlate_axes

SAMPLE_SCENARIO = Path(__file__).resolve().parent / "sample_data" / "battery_crisis.json"
ASPECT_CSV = Path(__file__).resolve().parent.parent / "dataset" / "ViSFD" / "ViSFD.csv"

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"
DATASET_LABELS = {
    "visfd": "ViSFD (review điện thoại)",
    "vietnamese-news": "Vietnamese-News",
    "visfd-e5": "ViSFD (review điện thoại, encoder E5)",
    "vietnamese-news-e5": "Vietnamese-News (encoder E5)",
    "uts-bank-e5": "UTS2017_Bank (phản ánh ngân hàng, encoder E5)",
}

st.set_page_config(page_title="S³ Topic Axis Explorer", layout="wide")


def discover_checkpoints() -> list[Path]:
    return sorted(ARTIFACTS_ROOT.glob("*/*/models/*.joblib"))


def pick_default(paths: list[Path]) -> Path | None:
    latest_only = [p for p in paths if p.name == "latest.joblib"]
    pool = latest_only or paths
    return max(pool, key=lambda p: p.stat().st_mtime) if pool else None


@st.cache_data(show_spinner=False)
def _n_topics_of(path_str: str, mtime: float) -> int | str:
    # mtime in the cache key so an overwritten file (e.g. latest.joblib) busts the cache
    try:
        return load_model(Path(path_str)).metadata.get("n_topics", "?")
    except Exception:
        return "?"


def describe(path: Path) -> str:
    # .../artifacts/<backend>/<dataset>/models/<file>.joblib
    backend, dataset = path.parts[-4], path.parts[-3]
    dataset_label = DATASET_LABELS.get(dataset, dataset)
    n_topics = _n_topics_of(str(path), path.stat().st_mtime)
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M %d/%m")
    marker = " · mới nhất" if path.name == "latest.joblib" else ""
    return f"{dataset_label} · {backend} · {n_topics} topics · {when}{marker}"


@st.cache_resource(show_spinner="Đang nạp checkpoint...")
def get_checkpoint(path_str: str) -> ModelCheckpoint:
    return load_model(Path(path_str))


@st.cache_resource(show_spinner="Đang tải encoder (lần đầu có thể mất một lúc)...")
def get_encoder(model_name: str, encoder_kind: str = "cafebert"):
    # encoder_kind picks the wrapper class (pooling differs); checkpoints saved
    # before this field existed default to cafebert, the only encoder back then.
    return ENCODERS.get(encoder_kind, ENCODERS["cafebert"])(model_name=model_name, batch_size=1)


@st.cache_data(show_spinner="Đang nạp toàn bộ từ vựng...")
def load_full_vocabulary(dataset_dir: str) -> tuple[list[str], np.ndarray] | tuple[None, None]:
    """vocabulary.json + word_embeddings.npy sit next to models/ -- same files
    cli.py wrote during training. Loading them lets us re-rank the FULL
    vocabulary against a checkpoint's axes, not just the top_n saved at
    training time (checkpoints only keep the top 10 words by default)."""
    directory = Path(dataset_dir)
    vocab_path = directory / "vocabulary.json"
    embeddings_path = directory / "word_embeddings.npy"
    if not vocab_path.exists() or not embeddings_path.exists():
        return None, None
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    return vocabulary, embeddings


def as_word_scores(words: list[str]) -> list[WordScore]:
    """Checkpoint fallback path only has plain words (no score was saved) --
    wrap them so downstream formatting can treat both cases uniformly."""
    return [(word, float("nan")) for word in words]


def render_axis_scores(
    scores: np.ndarray,
    topics: list[list[WordScore]],
    topics_negative: list[list[WordScore]],
    has_negative: bool,
    key_prefix: str,
) -> None:
    """Chart + table of a projected point's score on every axis -- shared by
    both the full-paragraph analysis and the keyword/concept probe below.
    Each word in topics/topics_negative carries its own combined-importance
    score (NaN when falling back to a checkpoint's plain saved word list)."""
    positive_words = [format_word_scores(words) for words in topics]
    negative_words = [format_word_scores(words) for words in topics_negative] if has_negative else ["" for _ in topics]
    top_pos_word = [words[0][0] if words else "?" for words in topics]
    top_neg_word = [words[0][0] if words else "?" for words in topics_negative] if has_negative else ["?" for _ in topics]
    df = pd.DataFrame(
        {
            "axis": list(range(len(scores))),
            "topic": [f"Topic {i}" for i in range(len(scores))],
            "score": scores,
            "positive_words": positive_words,
            "negative_words": negative_words,
            "top_pos_word": top_pos_word,
            "top_neg_word": top_neg_word,
        }
    )
    df["label"] = df.apply(lambda r: f"{r['axis']}: {r['top_pos_word']} / {r['top_neg_word']}", axis=1)
    df = df.sort_values("score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    top = df.iloc[0]
    if top["score"] >= 0:
        st.subheader(f"Trục nổi bật nhất: {top['topic']} -- nghiêng về cực dương")
        st.write(f"Từ khoá khớp: **{top['positive_words']}**")
        if top["negative_words"]:
            st.caption(f"Cực đối lập của trục này: {top['negative_words']}")
    else:
        st.subheader(f"Trục nổi bật nhất: {top['topic']} -- nghiêng về cực âm")
        st.write(f"Từ khoá khớp: **{top['negative_words'] or '(chưa có, model chưa lưu cực âm)'}**")
        st.caption(f"Cực đối lập của trục này: {top['positive_words']}")

    st.subheader("Điểm trên từng trục (cả hai cực)")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Điểm chiếu"),
            y=alt.Y("label:N", sort=alt.SortField("score", order="descending"), title=None),
            color=alt.condition(alt.datum.score > 0, alt.value("#2E7D32"), alt.value("#C62828")),
            tooltip=[
                alt.Tooltip("topic:N", title="Trục"),
                alt.Tooltip("score:Q", title="Điểm", format=".3f"),
                alt.Tooltip("positive_words:N", title="Cực dương"),
                alt.Tooltip("negative_words:N", title="Cực âm"),
            ],
        )
        .properties(height=max(220, 24 * len(df)))
    )
    st.altair_chart(chart, use_container_width=True, key=f"{key_prefix}_chart")

    st.subheader("Chi tiết theo từng trục")
    show_df = df[["topic", "score", "positive_words", "negative_words"]].rename(
        columns={"topic": "Trục", "score": "Điểm", "positive_words": "Cực dương (+)", "negative_words": "Cực âm (-)"}
    )
    st.dataframe(show_df, use_container_width=True, hide_index=True, key=f"{key_prefix}_table")


# --- Sidebar: chọn / nạp model -------------------------------------------------
st.sidebar.title("Model")
checkpoints = discover_checkpoints()

if not checkpoints:
    st.error(
        "Chưa có checkpoint nào trong `artifacts/*/*/models/`. "
        "Chạy training trước, ví dụ:\n\n"
        "`python -m s3_reproduction.cli --backend turftopic --dataset visfd --n-topics 20`"
    )
    st.stop()

checkpoints = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
default_path = pick_default(checkpoints)
labels = [describe(p) for p in checkpoints]
default_index = checkpoints.index(default_path) if default_path in checkpoints else 0

chosen_label = st.sidebar.selectbox(
    "Chọn checkpoint", labels, index=default_index,
    help="latest = lần train gần nhất cho backend/dataset đó; các mục còn lại đặt tên theo giờ:phút_ngày-tháng-năm.",
)
chosen_path = checkpoints[labels.index(chosen_label)]

if "active_checkpoint" not in st.session_state:
    st.session_state.active_checkpoint = str(default_path)

if st.sidebar.button("Load model", type="primary", use_container_width=True):
    st.session_state.active_checkpoint = str(chosen_path)
    st.cache_resource.clear()

ckpt = get_checkpoint(st.session_state.active_checkpoint)
meta = ckpt.metadata
dataset_dir = str(Path(st.session_state.active_checkpoint).parent.parent)
full_vocabulary, full_embeddings = load_full_vocabulary(dataset_dir)

st.sidebar.markdown("---")
st.sidebar.caption(f"Đang dùng: **{describe(Path(st.session_state.active_checkpoint))}**")
st.sidebar.metric("Số trục (n_topics)", meta.get("n_topics", len(ckpt.topics)))
col_a, col_b = st.sidebar.columns(2)
col_a.metric("Diversity", f"{meta.get('topic_diversity', 0):.3f}")
col_b.metric("Coherence", f"{meta.get('embedding_coherence', 0):.3f}")
st.sidebar.caption(f"Encoder: `{meta.get('encoder', 'uitnlp/CafeBERT')}`")
if isinstance(meta.get("documents"), int):
    st.sidebar.caption(f"Huấn luyện trên {meta['documents']:,} tài liệu")

train_timing = meta.get("timing") or {}
if train_timing:
    st.sidebar.caption(
        f"Thời gian lúc train: mã hoá {train_timing.get('embedding_seconds', 0):.1f}s · "
        f"fit model {train_timing.get('model_seconds', 0):.2f}s"
    )

if full_vocabulary is not None:
    st.sidebar.caption(f"Từ vựng đầy đủ: **{len(full_vocabulary):,} từ** -- model xếp hạng được hết, không chỉ top 10.")
    top_k = st.sidebar.slider(
        "Số từ hiển thị mỗi cực", min_value=5, max_value=min(100, len(full_vocabulary) // 2),
        value=20, step=5,
        help="Model chấm điểm toàn bộ từ vựng cho mỗi trục; đây chỉ là số từ hiển thị, không phải giới hạn của model. "
             "20 từ + điểm số đủ để suy ra ý nghĩa trục -- 6-10 từ thường quá chung chung.",
    )
    topics, topics_negative = rank_vocabulary(ckpt.ica, full_vocabulary, full_embeddings, top_k)
else:
    st.sidebar.caption(
        "Không tìm thấy `vocabulary.json`/`word_embeddings.npy` cạnh checkpoint này -- "
        f"chỉ hiện được top {meta.get('top_n', 10)} từ đã lưu sẵn lúc train (không kèm điểm)."
    )
    topics, topics_negative = [as_word_scores(t) for t in ckpt.topics], [as_word_scores(t) for t in ckpt.topics_negative]
has_negative = bool(topics_negative)

with st.sidebar.expander("Tất cả trục của model này (2 cực, kèm điểm)"):
    for i, words in enumerate(topics):
        negative = topics_negative[i] if has_negative and i < len(topics_negative) else []
        st.caption(f"**{i}** + {format_word_scores(words[:10])}")
        if negative:
            st.caption(f"　　－ {format_word_scores(negative[:10])}")

# --- Main: nhập văn bản, chiếu lên trục ----------------------------------------
st.title("Phân tích trục chủ đề (S³)")
st.write(
    "Nhập một đoạn văn tiếng Việt. Hệ thống mã hoá bằng đúng encoder đã dùng để train model đang chọn, "
    "rồi chiếu embedding đó lên các trục ngữ nghĩa đã học -- không train lại gì. "
    "Mỗi trục có hai cực (paper gọi cực thấp nhất là *negative definition* của topic)."
)

text = st.text_area("Đoạn văn cần phân tích", height=160, placeholder="Nhập văn bản ở đây...")
run = st.button("Phân tích", type="primary")

if run and text.strip():
    total_start = time.perf_counter()

    encoder = get_encoder(meta.get("encoder", "uitnlp/CafeBERT"), meta.get("encoder_kind", "cafebert"))
    embed_start = time.perf_counter()
    embedding = encoder.encode([text.strip()], "input")
    embedding_seconds = time.perf_counter() - embed_start

    model_start = time.perf_counter()
    scores = ckpt.ica.transform(embedding)[0]
    model_seconds = time.perf_counter() - model_start

    total_seconds = time.perf_counter() - total_start

    t1, t2, t3 = st.columns(3)
    t1.metric("Thời gian mã hoá", f"{embedding_seconds * 1000:.0f} ms")
    t2.metric("Thời gian chiếu trục (ICA)", f"{model_seconds * 1000:.0f} ms")
    t3.metric("Tổng thời gian", f"{total_seconds * 1000:.0f} ms")

    render_axis_scores(scores, topics, topics_negative, has_negative, key_prefix="doc")
elif run:
    st.warning("Nhập văn bản trước đã.")

# --- Xác định trục theo khái niệm / từ khoá ------------------------------------
st.markdown("---")
st.title("Xác định trục theo khái niệm")
st.write(
    "Thay vì đoán trục qua đọc từ khoá, kiểm tra trực tiếp: gộp vài từ khoá cùng nghĩa "
    "(vd. *pin, sạc, dung lượng pin, hết pin nhanh*) thành một vector khái niệm rồi chiếu lên "
    "các trục -- trục nào điểm cao nhất là trục khớp nhất với khái niệm đó."
)
keyword_text = st.text_input(
    "Từ khoá / cụm từ (cách nhau bằng dấu phẩy)",
    placeholder="pin, sạc, dung lượng pin, hết pin nhanh",
)
check = st.button("Kiểm tra trục", type="primary")

if check and keyword_text.strip():
    keywords = [k.strip() for k in keyword_text.split(",") if k.strip()]
    encoder = get_encoder(meta.get("encoder", "uitnlp/CafeBERT"), meta.get("encoder_kind", "cafebert"))
    keyword_embeddings = encoder.encode(keywords, "keywords")
    concept_embedding = keyword_embeddings.mean(axis=0, keepdims=True)
    concept_scores = ckpt.ica.transform(concept_embedding)[0]
    individual_scores = ckpt.ica.transform(keyword_embeddings)  # (n_keywords, n_topics)

    if len(keywords) > 1:
        best_axis_per_keyword = individual_scores.argmax(axis=1)
        agree_df = pd.DataFrame(
            {
                "Từ khoá": keywords,
                "Trục khớp nhất": [f"Topic {i}" for i in best_axis_per_keyword],
                "Điểm": individual_scores[range(len(keywords)), best_axis_per_keyword],
            }
        )
        n_agree = pd.Series(best_axis_per_keyword).nunique()
        if n_agree == 1:
            st.success(f"Cả {len(keywords)} từ khoá đều trỏ về cùng một trục -- khái niệm khá rõ ràng.")
        else:
            st.warning(
                f"{len(keywords)} từ khoá trỏ về {n_agree} trục khác nhau -- khái niệm có thể trải "
                "trên nhiều trục thay vì gói gọn trong một trục duy nhất."
            )
        st.dataframe(agree_df, use_container_width=True, hide_index=True)

    render_axis_scores(concept_scores, topics, topics_negative, has_negative, key_prefix="concept")
elif check:
    st.warning("Nhập ít nhất một từ khoá trước đã.")

# --- Giám sát & Cảnh báo --------------------------------------------------------
st.markdown("---")
st.title("Giám sát & Cảnh báo")
st.write(
    "Đổ một lô comment mới (chưa có nhãn) vào, so điểm trục của lô này với phân phối "
    "bình thường lúc train -- trục nào lệch nhiều độ lệch chuẩn là dấu hiệu bất thường. "
    "Không dùng AUC ở đây (AUC cần nhãn thật của chính lô đang xét, lô mới không có) -- "
    "nhãn hiển thị bên dưới chỉ là tra cứu lại kết quả hiệu chỉnh offline "
    "(`validate_visfd.py`, tính một lần trên dữ liệu ViSFD gốc), không tính lại mỗi lần."
)


@st.cache_data(show_spinner="Đang tính baseline & hiệu chỉnh nhãn (vài giây)...")
def get_monitor_setup(checkpoint_path: str, csv_path: str):
    ckpt_local = load_model(Path(checkpoint_path))
    dataset_dir_local = Path(checkpoint_path).parent.parent
    doc_embeddings = np.load(dataset_dir_local / "document_embeddings.npy")
    doc_topic = ckpt_local.ica.transform(doc_embeddings)
    baseline_local = compute_baseline(doc_topic)
    labels: dict[int, tuple[str, float]] = {}
    if csv_path:
        frame, aspects = build_aspect_frame(Path(csv_path))
        frame = frame.iloc[: len(doc_topic)].reset_index(drop=True)
        if len(frame) == len(doc_topic):
            results = correlate_axes(doc_topic, frame, aspects)
            for axis, rows in results.groupby("axis"):
                best = rows.loc[rows["auc"].idxmax()]
                if best["auc"] >= 0.65:
                    labels[int(axis)] = (best["aspect"], float(best["auc"]))
    return baseline_local, labels


dataset_name = Path(st.session_state.active_checkpoint).parts[-3]
can_label = dataset_name.startswith("visfd") and ASPECT_CSV.exists()
baseline, axis_labels = get_monitor_setup(
    st.session_state.active_checkpoint, str(ASPECT_CSV) if can_label else ""
)
if not can_label:
    st.caption(
        "Model đang chọn không phải ViSFD nên chưa có nhãn khía cạnh để đối chiếu -- "
        "vẫn phát hiện được bất thường, chỉ không gắn được tên khía cạnh."
    )

sample_available = SAMPLE_SCENARIO.exists()
use_sample = st.checkbox(
    "Dùng kịch bản mẫu (giả định: khủng hoảng pin phát nổ, 51 comment mô phỏng)",
    value=sample_available, disabled=not sample_available,
)
default_batch_text = (
    "\n".join(json.loads(SAMPLE_SCENARIO.read_text(encoding="utf-8"))) if use_sample and sample_available else ""
)
batch_text = st.text_area("Lô comment mới (mỗi dòng 1 comment)", value=default_batch_text, height=200)
z_threshold = st.slider("Ngưỡng cảnh báo (z-score)", min_value=2.0, max_value=6.0, value=3.0, step=0.5)
analyze_batch = st.button("Phân tích lô", type="primary")

if analyze_batch:
    batch_comments = [line.strip() for line in batch_text.splitlines() if line.strip()]
    if len(batch_comments) < 5:
        st.warning("Cần ít nhất 5 comment để so với baseline có ý nghĩa thống kê.")
    else:
        encoder = get_encoder(meta.get("encoder", "uitnlp/CafeBERT"), meta.get("encoder_kind", "cafebert"))
        with st.spinner(f"Đang mã hoá {len(batch_comments)} comment..."):
            batch_embeddings = encoder.encode(batch_comments, "batch")
            batch_topic = ckpt.ica.transform(batch_embeddings)
        z = batch_zscores(batch_topic, baseline)

        order = np.argsort(-np.abs(z))
        flagged = [int(a) for a in order if abs(z[a]) >= z_threshold]
        labeled_flagged = [a for a in flagged if a in axis_labels]

        if labeled_flagged:
            top_axis = labeled_flagged[0]
            aspect, auc = axis_labels[top_axis]
            st.error(
                f"CẢNH BÁO: khả năng cao đang có bất thường về **{aspect}** "
                f"(trục {top_axis}, z={z[top_axis]:.1f}, độ tin cậy nhãn AUC={auc:.2f})"
            )
        elif flagged:
            st.warning(f"Có {len(flagged)} trục lệch bất thường nhưng chưa trục nào có nhãn đủ tin cậy để gọi tên khía cạnh.")
        else:
            st.success("Không phát hiện bất thường vượt ngưỡng.")

        alert_rows = []
        for axis in flagged:
            aspect, auc = axis_labels.get(axis, ("(chưa rõ nghĩa)", None))
            words = ckpt.topics[axis][:6] if z[axis] > 0 else ckpt.topics_negative[axis][:6]
            alert_rows.append(
                {
                    "Trục": f"Topic {axis}{'+' if z[axis] > 0 else '-'}",
                    "z-score": float(z[axis]),
                    "Khía cạnh": aspect,
                    "AUC nhãn": auc,
                    "Từ khoá": ", ".join(words),
                }
            )
        if alert_rows:
            alerts_df = pd.DataFrame(alert_rows).sort_values("z-score", key=lambda s: s.abs(), ascending=False)
            alert_chart = (
                alt.Chart(alerts_df)
                .mark_bar()
                .encode(
                    x=alt.X("z-score:Q"),
                    y=alt.Y("Trục:N", sort=alt.SortField("z-score", order="descending"), title=None),
                    color=alt.condition(alt.datum["z-score"] > 0, alt.value("#2E7D32"), alt.value("#C62828")),
                    tooltip=["Trục", "z-score", "Khía cạnh", "AUC nhãn", "Từ khoá"],
                )
                .properties(height=max(160, 26 * len(alerts_df)))
            )
            st.altair_chart(alert_chart, use_container_width=True, key="monitor_chart")
            st.dataframe(alerts_df, use_container_width=True, hide_index=True, key="monitor_table")

# --- Định tuyến phòng ban (UTS2017_Bank) -----------------------------------------
bank_dataset_name = Path(st.session_state.active_checkpoint).parts[-3]
if bank_dataset_name.startswith("uts-bank"):
    st.markdown("---")
    st.title("Định tuyến câu hỏi -> phòng ban")
    st.write(
        "Nhập 1 câu hỏi/phản ánh của khách hàng -- hệ thống chiếu qua các trục đã được "
        "hiệu chỉnh với 14 nhãn khía cạnh ngân hàng thật (`validate_uts_bank.py`, offline, "
        "một lần) rồi gợi ý phòng ban khớp nhất. Đây chính là cơ chế server thông báo "
        "(`server/`) dùng để phát cảnh báo realtime -- xem `server/README.md`."
    )

    @st.cache_data(show_spinner="Đang hiệu chỉnh nhãn phòng ban (vài giây)...")
    def get_bank_label_axis(checkpoint_path: str):
        ckpt_local = load_model(Path(checkpoint_path))
        dataset_dir_local = Path(checkpoint_path).parent.parent
        doc_embeddings = np.load(dataset_dir_local / "document_embeddings.npy")
        doc_topic_local = ckpt_local.ica.transform(doc_embeddings)
        frame, labels = build_bank_label_frame()
        n = min(len(frame), len(doc_topic_local))
        frame = frame.iloc[:n].reset_index(drop=True)
        results = correlate_axes(doc_topic_local[:n], frame, labels)
        mapping = []
        for label in labels:
            rows = results[results["aspect"] == label]
            if rows.empty:
                continue
            best = rows.loc[rows["auc"].idxmax()]
            mapping.append(
                {
                    "label": label, "axis": int(best["axis"]),
                    "pole": 1 if best["mean_diff"] > 0 else -1,
                    "auc": float(best["auc"]),
                }
            )
        return mapping

    bank_mapping = get_bank_label_axis(st.session_state.active_checkpoint)
    with st.expander(f"{len(bank_mapping)} phòng ban đã hiệu chỉnh được (trong 14 nhãn gốc)"):
        map_df = pd.DataFrame(bank_mapping).sort_values("auc", ascending=False)
        map_df["Cực"] = map_df["pole"].map({1: "dương", -1: "âm"})
        st.dataframe(
            map_df[["label", "axis", "Cực", "auc"]].rename(
                columns={"label": "Phòng ban", "axis": "Trục", "auc": "AUC"}
            ),
            use_container_width=True, hide_index=True,
        )

    question_text = st.text_area(
        "Câu hỏi / phản ánh của khách hàng", height=100,
        placeholder="Thẻ VISA của em bị khoá mà không rõ lý do, mong hỗ trợ ạ.",
    )
    route_click = st.button("Định tuyến", type="primary")

    if route_click and question_text.strip():
        encoder = get_encoder(meta.get("encoder", "uitnlp/CafeBERT"), meta.get("encoder_kind", "cafebert"))
        with st.spinner("Đang mã hoá và chấm điểm..."):
            q_embedding = encoder.encode([question_text.strip()], "question")
            q_scores = ckpt.ica.transform(q_embedding)[0]

        route_rows = []
        for entry in bank_mapping:
            raw = q_scores[entry["axis"]] * entry["pole"]  # sign-adjusted: higher = more this department
            route_rows.append({"Phòng ban": entry["label"], "Điểm": float(raw), "AUC nhãn": entry["auc"]})
        route_df = pd.DataFrame(route_rows).sort_values("Điểm", ascending=False)

        top = route_df.iloc[0]
        st.success(f"Gợi ý phòng ban: **{top['Phòng ban']}** (điểm {top['Điểm']:.2f}, AUC nhãn {top['AUC nhãn']:.2f})")
        if len(route_df) > 1 and route_df.iloc[1]["Điểm"] > 0.5 * top["Điểm"] and top["Điểm"] > 0:
            st.caption(f"Điểm sát nhì: {route_df.iloc[1]['Phòng ban']} ({route_df.iloc[1]['Điểm']:.2f}) -- câu hỏi có thể liên quan cả 2 phòng ban.")

        route_chart = (
            alt.Chart(route_df)
            .mark_bar()
            .encode(
                x=alt.X("Điểm:Q"),
                y=alt.Y("Phòng ban:N", sort="-x", title=None),
                color=alt.condition(alt.datum["Điểm"] > 0, alt.value("#2E7D32"), alt.value("#C62828")),
                tooltip=["Phòng ban", "Điểm", "AUC nhãn"],
            )
            .properties(height=max(160, 26 * len(route_df)))
        )
        st.altair_chart(route_chart, use_container_width=True, key="route_chart")
        st.dataframe(route_df, use_container_width=True, hide_index=True, key="route_table")
    elif route_click:
        st.warning("Nhập câu hỏi trước đã.")
