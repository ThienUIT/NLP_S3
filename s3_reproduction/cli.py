from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .data import load_dataset
from .encoder import CafeBERTEncoder
from .metrics import embedding_coherence, topic_diversity
from .model import S3TopicModel
from .turftopic_backend import fit_turftopic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tái lập S3 với CafeBERT trên dữ liệu tiếng Việt")
    parser.add_argument("--dataset", choices=["visfd", "vietnamese-news", "all"], default="all")
    parser.add_argument(
        "--backend", choices=["turftopic", "custom"], default="turftopic",
        help="Dùng implementation chính thức của tác giả hoặc bản tự triển khai",
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


def run_dataset(name: str, args: argparse.Namespace, encoder: CafeBERTEncoder) -> None:
    limit = args.max_documents or None
    documents = load_dataset(name, args.dataset_root, limit, args.seed)
    vectorizer = CountVectorizer(
        lowercase=True, token_pattern=r"(?u)\b[^\W\d_][^\W_]+\b",
        min_df=args.min_df, max_df=args.max_df, max_features=args.max_features,
    )
    vectorizer.fit(documents)
    vocabulary = vectorizer.get_feature_names_out().tolist()
    document_embeddings = encoder.encode(documents, f"{name}: documents")
    word_embeddings = encoder.encode(vocabulary, f"{name}: vocabulary")
    destination = args.output_dir / args.backend / name
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "document_embeddings.npy", document_embeddings)
    np.save(destination / "word_embeddings.npy", word_embeddings)
    (destination / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8"
    )

    for n_topics in args.n_topics:
        if args.backend == "turftopic":
            model, topics = fit_turftopic(
                documents=documents,
                document_embeddings=document_embeddings,
                vocabulary=vocabulary,
                vocabulary_embeddings=word_embeddings,
                vectorizer=vectorizer,
                encoder=encoder,
                n_topics=n_topics,
                random_state=args.seed,
            )
            if args.top_n != 10:
                topics = [
                    [str(word) for word, _score in terms]
                    for _topic_id, terms in model.get_topics(top_k=args.top_n)
                ]
        else:
            model = S3TopicModel(n_topics=n_topics, random_state=args.seed).fit(document_embeddings)
            scores = model.word_scores(word_embeddings, method="combined")
            topics = model.top_words(scores, vocabulary, args.top_n)
        result = {
            "dataset": name, "backend": args.backend,
            "encoder": encoder.model_name, "pooling": "masked_mean",
            "documents": len(documents), "vocabulary_size": len(vocabulary),
            "n_topics": n_topics, "top_n": args.top_n, "topics": topics,
            "topic_diversity": topic_diversity(topics),
            "embedding_coherence": embedding_coherence(topics, vocabulary, word_embeddings),
        }
        (destination / f"topics_{n_topics}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    encoder = CafeBERTEncoder(batch_size=args.batch_size, max_length=args.max_length, device=args.device)
    names = ["visfd", "vietnamese-news"] if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args, encoder)


if __name__ == "__main__":
    main()
