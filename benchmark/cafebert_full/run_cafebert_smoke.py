#!/usr/bin/env python3
"""Run the 24 real-data CafeBERT smoke tests for Vietnamese topic modelling.

The matrix is 4 corpora × 6 configurations at seed=42 and k=10. It is a
configuration-validation run, not the final speed benchmark: shared CafeBERT
encoding is recorded separately from model fit time to expose caching effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import subprocess
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import requests
import torch
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
from turftopic import SemanticSignalSeparation
from umap import UMAP

try:
    from .paths import ROOT, SOURCES, SMOKE_RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    from paths import ROOT, SOURCES, SMOKE_RESULTS
RESULTS = SMOKE_RESULTS
CONFIG_PATH = ROOT / "cafebert_smoke_config.json"
TOPIC_PATTERN = re.compile(r"(?u)\b\w[\w-]+\b")
IT_CATEGORIES = ("Giai tri tin hoc", "Hackers va Virus", "San pham tin hoc moi")


@dataclass(frozen=True)
class CorpusBundle:
    name: str
    docs: list[str]
    ids: list[str]
    manifest: dict[str, Any]


def stable_hash(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def source_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "snapshot-not-available"


def get_json_with_retry(url: str, *, params: dict[str, Any] | None = None, attempts: int = 5) -> dict[str, Any]:
    """Request public dataset metadata with bounded exponential retry for transient 5xx."""
    retryable = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=45)
            if response.status_code in retryable:
                raise requests.HTTPError(f"HTTP {response.status_code} from {response.url}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Unable to retrieve {url} after {attempts} attempts: {last_error}")


def hf_news_rows() -> tuple[list[dict[str, Any]], str]:
    """Read all 858 rows through the public Dataset Server, retaining revision."""
    api = "https://huggingface.co/api/datasets/vanhai123/vietnamese-news-dataset"
    metadata = get_json_with_retry(api)
    revision = str(metadata.get("sha", "revision-not-returned"))
    endpoint = "https://datasets-server.huggingface.co/rows"
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = get_json_with_retry(
            endpoint,
            params={
                "dataset": "vanhai123/vietnamese-news-dataset",
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 100,
            },
        )
        chunk = payload.get("rows", [])
        rows.extend(chunk)
        if len(chunk) == 0 or len(rows) >= int(payload.get("num_rows_total", len(rows))):
            break
        offset += len(chunk)
    return rows, revision


def load_vietnamese_news() -> CorpusBundle:
    rows, revision = hf_news_rows()
    records = []
    for item in rows:
        row = item.get("row", item)
        text = normalize_text(row.get("content"))
        row_idx = item.get("row_idx", len(records))
        if len(text) >= 10:
            records.append({"doc_id": f"hf-vietnamese-news:{int(row_idx):04d}", "text": text})
    frame = pd.DataFrame(records).drop_duplicates("doc_id").sort_values("doc_id").reset_index(drop=True)
    ids = frame.doc_id.tolist()
    return CorpusBundle(
        name="vietnamese-news",
        docs=frame.text.tolist(),
        ids=ids,
        manifest={
            "corpus": "vietnamese-news",
            "language": "vi",
            "n_documents": len(frame),
            "document_ids_sha256": stable_hash(ids),
            "minimum_text_chars": 10,
            "source": {
                "repository": "https://huggingface.co/datasets/vanhai123/vietnamese-news-dataset",
                "revision": revision,
                "text_field": "content",
                "label_field_not_used_for_fit": "label",
                "license": "other (research/educational statement on dataset card)",
            },
        },
    )


def load_visfd(config: dict[str, Any]) -> CorpusBundle:
    archive = SOURCES / "UIT-ViSFD.zip"
    if not archive.exists():
        raise FileNotFoundError(f"UIT–ViSFD archive is missing: {archive}")
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive) as bundle:
        for split in ("Train.csv", "Dev.csv", "Test.csv"):
            member = next((name for name in bundle.namelist() if name == split or name.endswith(f"/{split}")), None)
            if member is None:
                raise ValueError(f"UIT–ViSFD archive has no {split}")
            with bundle.open(member) as handle:
                frame = pd.read_csv(handle)
            text_column = next((column for column in frame.columns if column.lower() in {"comment", "sentence", "text", "review"}), None)
            if text_column is None:
                raise ValueError(f"{split}: no known text column in {frame.columns.tolist()}")
            frames.append(pd.DataFrame({"doc_id": [f"uit-visfd:{split}:{i:05d}" for i in range(len(frame))], "text": frame[text_column].map(normalize_text)}))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.text.str.len() >= 10].drop_duplicates("doc_id")
    target = int(config["visfd"]["target_documents"])
    if len(frame) < target:
        raise ValueError(f"UIT–ViSFD has only {len(frame)} valid documents; expected at least {target}")
    sampled = frame.sample(n=target, random_state=int(config["visfd"]["sample_seed"])).sort_values("doc_id").reset_index(drop=True)
    ids = sampled.doc_id.tolist()
    return CorpusBundle(
        name="visfd",
        docs=sampled.text.tolist(),
        ids=ids,
        manifest={
            "corpus": "visfd",
            "language": "vi",
            "n_documents": len(sampled),
            "available_documents_after_filter": len(frame),
            "document_ids_sha256": stable_hash(ids),
            "minimum_text_chars": 10,
            "sample_seed": int(config["visfd"]["sample_seed"]),
            "source": {
                "repository": "https://github.com/LuongPhan/UIT-ViSFD",
                "snapshot_commit": "4b11ec2",
                "archive": archive.name,
                "splits": ["Train.csv", "Dev.csv", "Test.csv"],
                "labels_not_used_for_fit": True,
            },
        },
    )


def load_vi_medical() -> CorpusBundle:
    repository = SOURCES / "ViMedical_Disease"
    raw = pd.read_csv(repository / "ViMedical_Disease.csv")
    frame = pd.DataFrame({
        "doc_id": [f"vimedical:{i:05d}" for i in range(len(raw))],
        "text": raw["Question"].map(normalize_text),
    })
    frame = frame.loc[frame.text.str.len() >= 30].drop_duplicates("doc_id").sort_values("doc_id").reset_index(drop=True)
    ids = frame.doc_id.tolist()
    return CorpusBundle(
        name="vi-medical",
        docs=frame.text.tolist(),
        ids=ids,
        manifest={
            "corpus": "vi-medical",
            "language": "vi",
            "n_documents": len(frame),
            "available_raw_rows": len(raw),
            "document_ids_sha256": stable_hash(ids),
            "minimum_text_chars": 30,
            "source": {
                "repository": "https://github.com/PB3002/ViMedical_Disease",
                "snapshot_commit": source_commit(repository),
                "license": "CC BY-NC-SA 4.0",
                "text_field": "Question",
            },
        },
    )


def load_vntc_it() -> CorpusBundle:
    extracted = SOURCES / "VNTC_extracted"
    records: list[dict[str, str]] = []
    for split in ("train", "test"):
        for category in IT_CATEGORIES:
            folder = extracted / split / f"new {split}" / category
            for path in sorted(folder.glob("*.txt")):
                text = normalize_text(path.read_text(encoding="utf-16", errors="strict"))
                records.append({"doc_id": path.relative_to(extracted).as_posix(), "text": text})
    frame = pd.DataFrame(records)
    frame = frame.loc[frame.text.str.len() >= 150].drop_duplicates("doc_id").sort_values("doc_id").reset_index(drop=True)
    ids = frame.doc_id.tolist()
    return CorpusBundle(
        name="vntc-it",
        docs=frame.text.tolist(),
        ids=ids,
        manifest={
            "corpus": "vntc-it",
            "language": "vi",
            "n_documents": len(frame),
            "document_ids_sha256": stable_hash(ids),
            "minimum_text_chars": 150,
            "source": {
                "repository": "https://github.com/duyvuleo/VNTC",
                "snapshot_commit": source_commit(SOURCES / "VNTC"),
                "license": "MIT",
                "categories": list(IT_CATEGORIES),
                "text_field": "full article text",
            },
        },
    )


def load_corpus(name: str, config: dict[str, Any]) -> CorpusBundle:
    loaders = {"vietnamese-news": lambda: load_vietnamese_news(), "visfd": lambda: load_visfd(config), "vi-medical": load_vi_medical, "vntc-it": load_vntc_it}
    return loaders[name]()


class CafeBERTMeanEncoder:
    """Minimal encoder with SentenceTransformer-compatible ``encode`` method."""

    def __init__(self, checkpoint: str, max_length: int, batch_size: int, revision: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_source = os.environ.get("S3_CAFEBERT_CHECKPOINT_DIR", checkpoint)
        kwargs = {"revision": revision} if checkpoint_source == checkpoint and revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_source, use_fast=True, **kwargs)
        self.model = AutoModel.from_pretrained(checkpoint_source, **kwargs).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, sentences: list[str], **_: Any) -> np.ndarray:
        indexed_sentences = sorted(enumerate(sentences), key=lambda item: (len(item[1]), item[0]))
        vectors: list[np.ndarray | None] = [None] * len(sentences)
        total_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
        for batch_index, begin in enumerate(range(0, len(indexed_sentences), self.batch_size), start=1):
            indexed_batch = indexed_sentences[begin: begin + self.batch_size]
            original_indices, batch = zip(*indexed_batch)
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for original_index, vector in zip(original_indices, pooled.cpu().numpy().astype("float32")):
                vectors[original_index] = vector
            if batch_index == total_batches or batch_index % 25 == 0:
                print(json.dumps({
                    "embedding_progress": {
                        "completed_batches": batch_index,
                        "total_batches": total_batches,
                        "completed_documents": min(begin + len(batch), len(sentences)),
                        "total_documents": len(sentences),
                    }
                }), flush=True)
        if any(vector is None for vector in vectors):
            raise ValueError("CafeBERT batching did not restore every original document position")
        return np.vstack(vectors)


def cache_embeddings(encoder: CafeBERTMeanEncoder, bundle: CorpusBundle, config: dict[str, Any]) -> tuple[np.ndarray, float, Path]:
    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({"corpus": bundle.name, "ids": bundle.manifest["document_ids_sha256"], "encoder": config["encoder"]}, sort_keys=True).encode()).hexdigest()[:16]
    path = cache / f"{bundle.name}_{len(bundle.docs)}_{key}_cafebert_mean.npy"
    if path.exists():
        values = np.load(path)
        if values.shape[0] == len(bundle.docs) and np.isfinite(values).all():
            return values, 0.0, path
    start = time.perf_counter()
    values = encoder.encode(bundle.docs)
    elapsed = time.perf_counter() - start
    if values.shape[0] != len(bundle.docs) or not np.isfinite(values).all():
        raise ValueError(f"Invalid embeddings {values.shape}; finite={np.isfinite(values).all()}")
    np.save(path, values)
    return values, elapsed, path


def extract_topics(raw_topics: Any, n_topics: int, topn: int) -> list[list[str]]:
    entries = [raw_topics[key] for key in sorted(raw_topics) if key != -1] if isinstance(raw_topics, dict) else raw_topics
    topics: list[list[str]] = []
    for value in entries:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (list, tuple, np.ndarray)):
            value = value[1]
        words: list[str] = []
        for item in list(value)[:topn]:
            word = str(item[0] if isinstance(item, (tuple, list, np.ndarray)) else item)
            if word and word not in words:
                words.append(word)
        if words:
            topics.append(words)
    return (topics + [[] for _ in range(n_topics)])[:n_topics]


def make_word2vec(tokenized_docs: list[list[str]]) -> Word2Vec:
    return Word2Vec(tokenized_docs, min_count=1, seed=42, workers=1)


def wec_in(topics: list[list[str]], vectors: Any) -> float:
    values = []
    for topic in topics:
        pairs = [vectors.similarity(left, right) for left, right in combinations(topic, 2) if left in vectors and right in vectors]
        if pairs:
            values.append(float(np.mean(pairs)))
    return float(np.mean(values)) if values else float("nan")


def diversity(topics: list[list[str]], topn: int) -> float:
    words = [word for topic in topics for word in topic]
    return len(set(words)) / max(1, len(topics) * topn)


def c_npmi(topics: list[list[str]], tokenized_docs: list[list[str]], dictionary: Dictionary) -> float:
    from gensim.models import CoherenceModel
    usable = [[word for word in topic if word in dictionary.token2id] for topic in topics]
    usable = [topic for topic in usable if len(topic) >= 2]
    if not usable:
        return float("nan")
    return float(CoherenceModel(topics=usable, texts=tokenized_docs, dictionary=dictionary, coherence="c_npmi", processes=1).get_coherence())


def alpha_rate(topics: list[list[str]]) -> float:
    words = [word for topic in topics for word in topic]
    return sum(word.replace("_", "").isalpha() for word in words) / max(1, len(words))


def build_vectorizer(config: dict[str, Any], section: str = "lexical_tokenizer") -> CountVectorizer:
    kwargs = config[section]["CountVectorizer"]
    return CountVectorizer(min_df=int(kwargs["min_df"]), token_pattern=str(kwargs["token_pattern"]), lowercase=bool(kwargs["lowercase"]))


def run_model(name: str, docs: list[str], embeddings: np.ndarray, vectorizer: CountVectorizer, encoder: CafeBERTMeanEncoder, seed: int, n_topics: int, config: dict[str, Any]) -> list[list[str]]:
    topn = int(config["top_terms"])
    if name.startswith("s3_"):
        feature = name.removeprefix("s3_")
        model = SemanticSignalSeparation(n_components=n_topics, encoder=encoder, vectorizer=clone(vectorizer), random_state=seed, feature_importance=feature)
        model.fit(docs, embeddings=embeddings)
        return extract_topics(model.get_topics(), n_topics, topn)
    if name in {"lda", "nmf"}:
        local = clone(vectorizer)
        matrix = local.fit_transform(docs)
        if name == "lda":
            model = LatentDirichletAllocation(n_components=n_topics, random_state=seed).fit(matrix)
        else:
            model = NMF(n_components=n_topics, random_state=seed, init="nndsvda").fit(matrix)
        vocabulary = np.asarray(local.get_feature_names_out())
        return [vocabulary[np.argsort(-component)[:topn]].tolist() for component in model.components_]
    if name == "bertopic_kmeans":
        settings = config["bertopic"]
        bertopic_vectorizer = build_vectorizer(config, "bertopic_vectorizer")
        topic_model = BERTopic(
            embedding_model=None,
            vectorizer_model=bertopic_vectorizer,
            umap_model=UMAP(n_neighbors=int(settings["n_neighbors"]), n_components=int(settings["n_components"]), min_dist=float(settings["min_dist"]), metric=str(settings["metric"]), random_state=seed),
            hdbscan_model=KMeans(n_clusters=n_topics, random_state=seed, n_init=int(settings["n_init"])),
            calculate_probabilities=False,
            verbose=False,
        )
        topic_model.fit_transform(docs, embeddings=embeddings)
        return extract_topics(topic_model.get_topics(), n_topics, topn)
    raise ValueError(f"Unsupported model {name}")


def environment() -> dict[str, Any]:
    import bertopic
    import gensim
    import sklearn
    import transformers
    import turftopic
    import umap
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "turftopic": getattr(turftopic, "__version__", "unreported"),
        "bertopic": bertopic.__version__,
        "umap": umap.__version__,
        "sklearn": sklearn.__version__,
        "gensim": gensim.__version__,
    }


def completed_keys(path: Path) -> set[tuple[str, str, int, int]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    valid = frame.loc[frame.status.eq("ok"), ["corpus", "model", "seed", "n_topics"]]
    return set(map(tuple, valid.itertuples(index=False, name=None)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", nargs="+", default=None, choices=("vietnamese-news", "visfd", "vi-medical", "vntc-it"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    corpora = args.corpora or list(config["corpora"])
    seed, n_topics = int(config["seed"]), int(config["n_topics"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(environment(), ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    bundles = {name: load_corpus(name, config) for name in corpora}
    for name, bundle in bundles.items():
        (RESULTS / f"{name}_manifest.json").write_text(json.dumps(bundle.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"prepared": name, "n_documents": len(bundle.docs), "document_ids_sha256": bundle.manifest["document_ids_sha256"]}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    output = RESULTS / "smoke_results.csv"
    rows = pd.read_csv(output).to_dict(orient="records") if output.exists() else []
    done = set() if args.force else completed_keys(output)
    encoder_config = config["encoder"]
    model_started = time.perf_counter()
    encoder = CafeBERTMeanEncoder(str(encoder_config["checkpoint"]), int(encoder_config["max_length"]), int(encoder_config["batch_size"]), str(encoder_config.get("revision", "")) or None)
    model_load_seconds = time.perf_counter() - model_started
    all_topics_path = RESULTS / "smoke_topics.json"
    all_topics = json.loads(all_topics_path.read_text(encoding="utf-8")) if all_topics_path.exists() else {}
    for name, bundle in bundles.items():
        embeddings, embedding_seconds, embedding_path = cache_embeddings(encoder, bundle, config)
        vectorizer = build_vectorizer(config)
        vectorizer.fit(bundle.docs)
        analyzer = vectorizer.build_analyzer()
        tokenized = [analyzer(document) for document in bundle.docs]
        dictionary = Dictionary(tokenized)
        word_vectors = make_word2vec(tokenized)
        for model_name in config["models"]:
            key = (name, model_name, seed, n_topics)
            if key in done:
                print(f"SKIP completed {key}", flush=True)
                continue
            row: dict[str, Any] = {
                "corpus": name,
                "model": model_name,
                "seed": seed,
                "n_topics": n_topics,
                "n_documents": len(bundle.docs),
                "document_ids_sha256": bundle.manifest["document_ids_sha256"],
                "shared_encoder_load_seconds": model_load_seconds,
                "shared_embedding_seconds": embedding_seconds,
                "embedding_cache": str(embedding_path),
                "status": "failed",
            }
            try:
                started = time.perf_counter()
                topics = run_model(model_name, bundle.docs, embeddings, vectorizer, encoder, seed, n_topics, config)
                fit_seconds = time.perf_counter() - started
                metric_started = time.perf_counter()
                row.update({
                    "fit_seconds": fit_seconds,
                    "documents_per_second_fit": len(bundle.docs) / fit_seconds if fit_seconds else float("nan"),
                    "wec_in": wec_in(topics, word_vectors.wv),
                    "topic_diversity": diversity(topics, int(config["top_terms"])),
                    "c_npmi": c_npmi(topics, tokenized, dictionary),
                    "alphabetic_term_rate": alpha_rate(topics),
                    "metric_seconds": time.perf_counter() - metric_started,
                    "topic_count_returned": len(topics),
                    "min_topic_term_count": min((len(topic) for topic in topics), default=0),
                    "status": "ok",
                    "error": "",
                })
                values = [row["wec_in"], row["topic_diversity"], row["c_npmi"]]
                if not all(np.isfinite(values)) or row["topic_count_returned"] != n_topics or row["min_topic_term_count"] < 2:
                    raise ValueError(f"Invalid smoke output: finite={np.isfinite(values).tolist()}, topics={row['topic_count_returned']}, min_terms={row['min_topic_term_count']}")
                all_topics[f"{name}|{model_name}|{seed}|{n_topics}"] = topics
            except Exception as exc:  # Keep an auditable failed row, then continue the matrix.
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=5)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output, index=False)
            all_topics_path.write_text(json.dumps(all_topics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
