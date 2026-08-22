#!/usr/bin/env python3
"""Resumable full-corpus CafeBERT topic-model benchmark.

The runner preserves the validated smoke-test data loaders and configuration,
records representation and fit stages separately, and never converts a cache hit
into a zero-cost timing observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

try:
    from . import run_cafebert_smoke as base
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    import run_cafebert_smoke as base
    from paths import ROOT, RESULTS
CONFIG_PATH = ROOT / "cafebert_full_config.json"


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def parse_csv_set(value: str | None, allowed: list[Any]) -> list[Any]:
    if value is None:
        return list(allowed)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    converted: list[Any] = []
    for item in requested:
        candidate: Any = int(item) if isinstance(allowed[0], int) else item
        if candidate not in allowed:
            raise ValueError(f"Unsupported selection {candidate!r}; allowed={allowed}")
        converted.append(candidate)
    return converted


def completed_keys(path: Path, experiment_hash: str) -> set[tuple[str, str, int, int, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    columns = ["corpus", "model", "seed", "n_topics", "config_sha256"]
    valid = frame.loc[frame.status.eq("ok") & frame.config_sha256.eq(experiment_hash), columns]
    return set(map(tuple, valid.itertuples(index=False, name=None)))


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def ordered_models(models: list[str], seed: int, n_topics: int) -> list[str]:
    offset = (seed + n_topics // 10) % len(models)
    return models[offset:] + models[:offset]


def representation_for_corpus(
    bundle: base.CorpusBundle,
    encoder: base.CafeBERTMeanEncoder,
    config: dict[str, Any],
) -> tuple[np.ndarray, float, Path]:
    cache_dir = RESULTS / "representation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "corpus": bundle.name,
        "document_ids_sha256": bundle.manifest["document_ids_sha256"],
        "encoder": config["encoder"],
    }
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    array_path = cache_dir / f"{bundle.name}_{len(bundle.docs)}_{key}.npy"
    seconds_path = cache_dir / f"{bundle.name}_{len(bundle.docs)}_{key}.json"
    if array_path.exists() and seconds_path.exists():
        vectors = np.load(array_path)
        metadata = json.loads(seconds_path.read_text(encoding="utf-8"))
        if vectors.shape[0] == len(bundle.docs) and np.isfinite(vectors).all():
            return vectors, float(metadata["cold_representation_seconds"]), array_path
    started = time.perf_counter()
    vectors = encoder.encode(bundle.docs)
    elapsed = time.perf_counter() - started
    if vectors.shape[0] != len(bundle.docs) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid CafeBERT embeddings for {bundle.name}: shape={vectors.shape}")
    np.save(array_path, vectors)
    seconds_path.write_text(json.dumps({"cold_representation_seconds": elapsed, "fingerprint": fingerprint}, ensure_ascii=False, indent=2), encoding="utf-8")
    return vectors, elapsed, array_path


def lexical_representation(docs: list[str], config: dict[str, Any]) -> tuple[Any, Any, list[list[str]], Dictionary, Any, float]:
    started = time.perf_counter()
    vectorizer = base.build_vectorizer(config)
    matrix = vectorizer.fit_transform(docs)
    elapsed = time.perf_counter() - started
    analyzer = vectorizer.build_analyzer()
    tokenized = [analyzer(document) for document in docs]
    dictionary = Dictionary(tokenized)
    word2vec = base.make_word2vec(tokenized)
    return vectorizer, matrix, tokenized, dictionary, word2vec, elapsed


def run_lexical_model(name: str, matrix: Any, vectorizer: Any, seed: int, n_topics: int, topn: int) -> list[list[str]]:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    if name == "lda":
        model = LatentDirichletAllocation(n_components=n_topics, random_state=seed).fit(matrix)
    elif name == "nmf":
        model = NMF(n_components=n_topics, random_state=seed, init="nndsvda").fit(matrix)
    else:
        raise ValueError(name)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    return [vocabulary[np.argsort(-component)[:topn]].tolist() for component in model.components_]


def validate_row(row: dict[str, Any], expected_topics: int, top_terms: int) -> None:
    metrics = [row["wec_in"], row["topic_diversity"], row["c_npmi"], row["pipeline_seconds"], row["fit_seconds"], row["metric_seconds"]]
    if not all(np.isfinite(metrics)) or row["topic_count_returned"] != expected_topics or row["min_topic_term_count"] < top_terms:
        raise ValueError(f"Invalid output: finite={np.isfinite(metrics).tolist()}, topics={row['topic_count_returned']}, min_terms={row['min_topic_term_count']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", help="Comma-separated selected seeds; defaults to the primary seed list.")
    parser.add_argument("--corpora", help="Comma-separated corpus ids.")
    parser.add_argument("--topic-counts", help="Comma-separated k values.")
    parser.add_argument("--force", action="store_true", help="Rerun rows even if an OK row exists for the same key.")
    parser.add_argument("--dedupe-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_hash = config_hash(config)
    seeds = parse_csv_set(args.seeds, config["seeds_sensitivity"] if args.seeds else config["seeds_primary"])
    corpora = parse_csv_set(args.corpora, config["corpora"])
    topic_counts = parse_csv_set(args.topic_counts, config["topic_counts"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS / "full_results.csv"
    topic_path = RESULTS / "full_topics.json"
    existing_rows = pd.read_csv(result_path).to_dict(orient="records") if result_path.exists() else []

    if args.dedupe_only:
        frame = pd.DataFrame(existing_rows)
        keys = ["corpus", "model", "seed", "n_topics", "config_sha256"]
        if not frame.empty:
            frame = frame.drop_duplicates(keys, keep="last")
        frame.to_csv(result_path, index=False)
        print(json.dumps({"deduplicated_rows": len(frame), "keys": keys}), flush=True)
        return

    done = set() if args.force else completed_keys(result_path, experiment_hash)
    (RESULTS / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "experiment_contract.json").write_text((ROOT / "experiment_contract.json").read_text(encoding="utf-8"), encoding="utf-8")
    hardware = base.environment() | {"config_sha256": experiment_hash, "started_at_unix": time.time()}
    (RESULTS / "environment.json").write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus_load_started = time.perf_counter()
    bundles = {name: base.load_corpus(name, config) for name in corpora}
    corpus_load_seconds = time.perf_counter() - corpus_load_started
    for name, bundle in bundles.items():
        (RESULTS / f"{name}_manifest.json").write_text(json.dumps(bundle.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"prepared": name, "n_documents": len(bundle.docs), "document_ids_sha256": bundle.manifest["document_ids_sha256"]}, ensure_ascii=False), flush=True)

    encoder_started = time.perf_counter()
    encoder = base.CafeBERTMeanEncoder(str(config["encoder"]["checkpoint"]), int(config["encoder"]["max_length"]), int(config["encoder"]["batch_size"]), str(config["encoder"].get("revision", "")) or None)
    encoder_load_seconds = time.perf_counter() - encoder_started
    topics_by_key = json.loads(topic_path.read_text(encoding="utf-8")) if topic_path.exists() else {}

    for corpus_name, bundle in bundles.items():
        embeddings, cafebert_rep_seconds, embedding_path = representation_for_corpus(bundle, encoder, config)
        lexical_vectorizer, lexical_matrix, tokenized, dictionary, word2vec, lexical_rep_seconds = lexical_representation(bundle.docs, config)
        lexical_path = RESULTS / "representation_cache" / f"{corpus_name}_{len(bundle.docs)}_{bundle.manifest['document_ids_sha256'][:12]}_lexical.npz"
        lexical_path.parent.mkdir(parents=True, exist_ok=True)
        if not lexical_path.exists():
            from scipy.sparse import save_npz
            save_npz(lexical_path, lexical_matrix)
        for seed in seeds:
            for n_topics in topic_counts:
                for model_name in ordered_models(list(config["models"]), seed, n_topics):
                    key = (corpus_name, model_name, seed, n_topics, experiment_hash)
                    if key in done:
                        print(json.dumps({"skip_completed": key}), flush=True)
                        continue
                    uses_cafebert = model_name.startswith("s3_") or model_name == "bertopic_kmeans"
                    representation_seconds = cafebert_rep_seconds if uses_cafebert else lexical_rep_seconds
                    representation_path = str(embedding_path if uses_cafebert else lexical_path)
                    row: dict[str, Any] = {
                        "experiment_id": config["experiment_id"], "config_sha256": experiment_hash,
                        "corpus": corpus_name, "model": model_name, "seed": seed, "n_topics": n_topics,
                        "n_documents": len(bundle.docs), "document_ids_sha256": bundle.manifest["document_ids_sha256"],
                        "ingest_preprocess_seconds_shared": corpus_load_seconds / len(bundles),
                        "encoder_model_load_seconds_shared": encoder_load_seconds if uses_cafebert else 0.0,
                        "representation_seconds_cold_reference": representation_seconds,
                        "representation_reference_path": representation_path,
                        "representation_kind": "cafebert_mean_encode" if uses_cafebert else "countvectorizer_fit_transform",
                        "fit_seconds": float("nan"), "pipeline_seconds": float("nan"), "total_cold_seconds": float("nan"),
                        "metric_seconds": float("nan"), "status": "failed", "error": "",
                    }
                    try:
                        fit_started = time.perf_counter()
                        if model_name in {"lda", "nmf"}:
                            topics = run_lexical_model(model_name, lexical_matrix, lexical_vectorizer, seed, n_topics, int(config["top_terms"]))
                        else:
                            topics = base.run_model(model_name, bundle.docs, embeddings, lexical_vectorizer, encoder, seed, n_topics, config)
                        row["fit_seconds"] = time.perf_counter() - fit_started
                        row["pipeline_seconds"] = representation_seconds + row["fit_seconds"]
                        row["total_cold_seconds"] = row["ingest_preprocess_seconds_shared"] + row["encoder_model_load_seconds_shared"] + row["pipeline_seconds"]
                        metric_started = time.perf_counter()
                        row.update({
                            "wec_in": base.wec_in(topics, word2vec.wv),
                            "topic_diversity": base.diversity(topics, int(config["top_terms"])),
                            "c_npmi": base.c_npmi(topics, tokenized, dictionary),
                            "alphabetic_term_rate": base.alpha_rate(topics),
                            "metric_seconds": time.perf_counter() - metric_started,
                            "topic_count_returned": len(topics),
                            "min_topic_term_count": min((len(topic) for topic in topics), default=0),
                            "status": "ok",
                        })
                        validate_row(row, n_topics, int(config["top_terms"]))
                        topics_by_key[f"{corpus_name}|{model_name}|{seed}|{n_topics}|{experiment_hash[:12]}"] = topics
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["traceback"] = traceback.format_exc(limit=7)
                    existing_rows.append(row)
                    write_table(existing_rows, result_path)
                    topic_path.write_text(json.dumps(topics_by_key, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
