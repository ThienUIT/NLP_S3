from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer

from .checkpoint import save_latest, save_model
from .data import load_dataset
from .encoder import ENCODERS
from .metrics import embedding_coherence, topic_diversity
from .model import S3TopicModel
from .turftopic_backend import fit_turftopic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tái lập S3 với CafeBERT trên dữ liệu tiếng Việt")
    parser.add_argument(
        "--dataset", choices=["visfd", "vietnamese-news", "vietnamese-curated", "uts-bank", "all"], default="all",
        help="'all' vẫn chỉ chạy visfd+vietnamese-news; vietnamese-curated (12.17M doc, "
             "tải từ HF Hub) và uts-bank (2.471 phản ánh ngân hàng, 14 nhãn khía cạnh thật) "
             "phải chọn riêng",
    )
    parser.add_argument(
        "--backend", choices=["turftopic", "custom"], default="turftopic",
        help="Dùng implementation chính thức của tác giả hoặc bản tự triển khai",
    )
    parser.add_argument(
        "--encoder", choices=list(ENCODERS.keys()), default="cafebert",
        help="cafebert (mặc định, uitnlp/CafeBERT, tự chế masked-mean pooling) hoặc "
             "e5 (sentence-transformer thật -- dùng --encoder-model để đổi sang model khác, "
             "vd BAAI/bge-m3, paraphrase-multilingual-mpnet-base-v2, bkai-foundation-models/vietnamese-bi-encoder)",
    )
    parser.add_argument(
        "--encoder-model", default=None,
        help="Chỉ áp dụng khi --encoder e5: tên model sentence-transformers bất kỳ trên HF Hub "
             "(mặc định intfloat/multilingual-e5-base nếu bỏ trống)",
    )
    parser.add_argument(
        "--encoder-prefix", default=None,
        help="Tiền tố thêm trước mỗi văn bản trước khi encode (mặc định: tự suy ra -- "
             "\"query: \" nếu tên model có chữ e5, ngược lại để trống)",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-topics", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-documents", type=int, default=20000,
                        help="0 dùng toàn bộ; mặc định 20k để Vietnamese-News khả thi")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_dataset(name: str, args: argparse.Namespace, encoder) -> None:
    limit = args.max_documents or None
    documents = load_dataset(name, args.dataset_root, limit, args.seed)

    # Vietnamese is written syllable-by-syllable with spaces, so a plain
    # whitespace tokenizer splits multi-syllable words (e.g. "trầy xước",
    # "bảo hành") into meaningless separate syllables. Segment first so the
    # vectorizer's vocabulary candidates are real words, joined with "_"
    # (e.g. "trầy_xước") -- token_pattern below keeps "_" as a word character
    # instead of treating it as a delimiter.
    segment_start = time.perf_counter()
    segmented_documents = [ViTokenizer.tokenize(doc) for doc in documents]
    segment_seconds = time.perf_counter() - segment_start

    vectorizer = CountVectorizer(
        lowercase=True, token_pattern=r"(?u)\b[^\W\d][^\W]*\b",
        min_df=args.min_df, max_df=args.max_df, max_features=args.max_features,
    )
    vectorizer.fit(segmented_documents)
    vocabulary = vectorizer.get_feature_names_out().tolist()  # e.g. "trầy_xước"

    embed_start = time.perf_counter()
    document_embeddings = encoder.encode(documents, f"{name}: documents")
    vocabulary_text = [word.replace("_", " ") for word in vocabulary]
    word_embeddings = encoder.encode(vocabulary_text, f"{name}: vocabulary")
    embedding_seconds = time.perf_counter() - embed_start

    # Different encoders (and different underlying --encoder-model choices
    # within the same kind, e.g. e5-base vs bge-m3) produce differently-shaped,
    # non-comparable embeddings -- isolate their output so a run can never
    # overwrite (or silently corrupt, via a dimension mismatch) another
    # encoder's cached document/word embeddings that existing checkpoints
    # still depend on. Default model per kind keeps its existing folder name
    # (visfd, visfd-e5) for continuity; a non-default --encoder-model gets an
    # extra slug so it can't collide with either.
    default_model_names = {"cafebert": "uitnlp/CafeBERT", "e5": "intfloat/multilingual-e5-base"}
    is_default_model = encoder.model_name == default_model_names.get(args.encoder)
    if args.encoder == "cafebert" and is_default_model:
        dataset_dir_name = name
    elif is_default_model:
        dataset_dir_name = f"{name}-{args.encoder}"
    else:
        slug = encoder.model_name.rsplit("/", 1)[-1].lower().replace("_", "-")
        dataset_dir_name = f"{name}-{args.encoder}-{slug}"
    destination = args.output_dir / args.backend / dataset_dir_name
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "document_embeddings.npy", document_embeddings)
    np.save(destination / "word_embeddings.npy", word_embeddings)
    (destination / "vocabulary.json").write_text(
        json.dumps(vocabulary_text, ensure_ascii=False), encoding="utf-8"
    )

    model_seconds_total = 0.0
    for n_topics in args.n_topics:
        model_start = time.perf_counter()
        if args.backend == "turftopic":
            model, topics, topics_negative = fit_turftopic(
                documents=segmented_documents,
                document_embeddings=document_embeddings,
                vocabulary=vocabulary,
                vocabulary_embeddings=word_embeddings,
                vectorizer=vectorizer,
                encoder=encoder,
                n_topics=n_topics,
                random_state=args.seed,
                top_n=args.top_n,
            )
        else:
            model = S3TopicModel(n_topics=n_topics, random_state=args.seed).fit(document_embeddings)
            scores = model.word_scores(word_embeddings, method="combined")
            topics = model.top_words(scores, vocabulary, args.top_n, positive=True)
            topics_negative = model.top_words(scores, vocabulary, args.top_n, positive=False)
        model_seconds = time.perf_counter() - model_start
        model_seconds_total += model_seconds

        # Metrics need topics/vocabulary in the same ("_"-joined) form the
        # embeddings were indexed by; only convert to readable text after.
        diversity = topic_diversity(topics)
        coherence = embedding_coherence(topics, vocabulary, word_embeddings)
        topics_readable = [[w.replace("_", " ") for w in words] for words in topics]
        topics_negative_readable = [[w.replace("_", " ") for w in words] for words in topics_negative]

        result = {
            "dataset": name, "backend": args.backend,
            "encoder": encoder.model_name, "encoder_kind": args.encoder,
            "pooling": "masked_mean" if args.encoder == "cafebert" else "sentence-transformer",
            "documents": len(documents), "max_documents": args.max_documents, "seed": args.seed,
            "vocabulary_size": len(vocabulary),
            "n_topics": n_topics, "top_n": args.top_n,
            "topics": topics_readable, "topics_negative": topics_negative_readable,
            "topic_diversity": diversity,
            "embedding_coherence": coherence,
            "timing": {
                "segment_seconds": round(segment_seconds, 3),
                "embedding_seconds": round(embedding_seconds, 3),
                "model_seconds": round(model_seconds, 3),
                "total_seconds": round(segment_seconds + embedding_seconds + model_seconds, 3),
            },
        }
        (destination / f"topics_{n_topics}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{name}] n_topics={n_topics}  tách từ={segment_seconds:.1f}s  mã hoá={embedding_seconds:.1f}s  "
            f"model={model_seconds:.2f}s  tổng={segment_seconds + embedding_seconds + model_seconds:.1f}s"
        )

        # Persist the fitted decomposition (small: a few matrices, no encoder weights)
        # so a later session or the Streamlit demo can reuse it without re-encoding.
        ica = model.decomposition if args.backend == "turftopic" else model.ica_
        checkpoint_metadata = {k: v for k, v in result.items() if k not in ("topics", "topics_negative")}
        models_dir = destination / "models"
        saved_to = save_model(
            ica=ica, vocabulary=vocabulary_text, topics=topics_readable, topics_negative=topics_negative_readable,
            metadata=checkpoint_metadata, output_dir=models_dir,
        )
        save_latest(
            ica=ica, vocabulary=vocabulary_text, topics=topics_readable, topics_negative=topics_negative_readable,
            metadata=checkpoint_metadata, output_dir=models_dir,
        )
        print(f"Saved model checkpoint to {saved_to} (and {models_dir / 'latest.joblib'})")

    print(
        f"[{name}] xong: tách từ {segment_seconds:.1f}s + mã hoá {embedding_seconds:.1f}s (một lần) + "
        f"{len(args.n_topics)} lần fit model tổng {model_seconds_total:.1f}s "
        f"= {segment_seconds + embedding_seconds + model_seconds_total:.1f}s cho cả dataset này"
    )


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    encoder_kwargs = {"batch_size": args.batch_size, "max_length": args.max_length, "device": args.device}
    if args.encoder_model:
        encoder_kwargs["model_name"] = args.encoder_model
    if args.encoder == "e5" and args.encoder_prefix is not None:
        encoder_kwargs["prefix"] = args.encoder_prefix
    encoder = ENCODERS[args.encoder](**encoder_kwargs)
    names = ["visfd", "vietnamese-news"] if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args, encoder)
    print(f"Tổng thời gian toàn bộ lần chạy (bao gồm load encoder): {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()
