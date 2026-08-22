#!/usr/bin/env python3
"""Create a compact, audited CafeBERT artifact for the S³ Signal Workbench."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


try:
    from .paths import ROOT, RESULTS
except ImportError:  # pragma: no cover - direct invocation convenience
    from paths import ROOT, RESULTS
CONFIG_PATH = ROOT / "cafebert_full_config.json"
CSV_PATH = RESULTS / "full_results.csv"
TOPICS_PATH = RESULTS / "full_topics.json"
AUDIT_PATH = RESULTS / "FULL_MULTISEED_AUDIT.md"
OUT = RESULTS / "s3-cafebert-full-corpus-workbench.json"

CORPORA = ["vietnamese-news", "visfd", "vi-medical", "vntc-it"]
SEEDS = [11, 29, 42, 47]
TOPIC_COUNTS = [10, 20, 30, 40, 50]
MODELS = ["s3_axial", "s3_angular", "s3_combined", "lda", "nmf", "bertopic_kmeans"]
MODEL_META = {
    "s3_axial": ("S³ axial", "s3", "axial"),
    "s3_angular": ("S³ angular", "s3", "angular"),
    "s3_combined": ("S³ combined", "s3", "combined"),
    "lda": ("LDA", "lda", "default"),
    "nmf": ("NMF", "nmf", "default"),
    "bertopic_kmeans": ("BERTopic + UMAP + KMeans", "bertopic_kmeans", "umap_kmeans"),
}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def corpus_meta() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for corpus in CORPORA:
        manifest = json.loads((RESULTS / f"{corpus}_manifest.json").read_text(encoding="utf-8"))
        source = manifest["source"]
        records.append(
            {
                "id": corpus,
                "name": {
                    "vietnamese-news": "Vietnamese-news",
                    "visfd": "UIT-ViSFD",
                    "vi-medical": "ViMedical Disease",
                    "vntc-it": "VNTC — CNTT",
                }[corpus],
                "language": manifest["language"],
                "documents": int(manifest["n_documents"]),
                "available_documents_after_filter": int(manifest.get("available_documents_after_filter", manifest.get("available_raw_rows", manifest["n_documents"]))),
                "min_text_chars": int(manifest["minimum_text_chars"]),
                "id_hash": manifest["document_ids_sha256"],
                "source": source["repository"],
                "source_commit": source.get("snapshot_commit", source.get("revision", "N/A")),
                "license": source.get("license", "Không công bố trong manifest"),
                "sample_seed": manifest.get("sample_seed"),
            }
        )
    return records


def row_payload(row: pd.Series) -> dict[str, object]:
    method, model, variant = MODEL_META[str(row.model)]
    return {
        "corpus": str(row.corpus),
        "method": method,
        "model": model,
        "variant": variant,
        "nTopics": int(row.n_topics),
        "wecIn": float(row.wec_in),
        "topicDiversity": float(row.topic_diversity),
        "cNpmi": float(row.c_npmi),
        "fitSeconds": float(row.fit_seconds),
        "pipelineSeconds": float(row.pipeline_seconds),
        "totalColdSeconds": float(row.total_cold_seconds),
        "metricSeconds": float(row.metric_seconds),
        "alphabeticTermRate": float(row.alphabetic_term_rate),
        "documentIdSha256": str(row.document_ids_sha256),
        "representationKind": str(row.representation_kind),
    }


def summary_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (corpus, model, n_topics), group in frame.groupby(["corpus", "model", "n_topics"], sort=True):
        method, ui_model, variant = MODEL_META[str(model)]
        rows.append(
            {
                "corpus": str(corpus),
                "method": method,
                "model": ui_model,
                "variant": variant,
                "nTopics": int(n_topics),
                "wecInMean": float(group.wec_in.mean()),
                "wecInSd": float(group.wec_in.std(ddof=1)),
                "diversityMean": float(group.topic_diversity.mean()),
                "diversitySd": float(group.topic_diversity.std(ddof=1)),
                "cNpmiMean": float(group.c_npmi.mean()),
                "cNpmiSd": float(group.c_npmi.std(ddof=1)),
                "fitSecondsMean": float(group.fit_seconds.mean()),
                "fitSecondsSd": float(group.fit_seconds.std(ddof=1)),
                "pipelineSecondsMean": float(group.pipeline_seconds.mean()),
                "pipelineSecondsSd": float(group.pipeline_seconds.std(ddof=1)),
                "totalColdSecondsMean": float(group.total_cold_seconds.mean()),
                "metricSecondsMean": float(group.metric_seconds.mean()),
                "alphabeticTermRateMean": float(group.alphabetic_term_rate.mean()),
                "runCount": int(group.shape[0]),
            }
        )
    return rows


def topic_payload(frame: pd.DataFrame, raw_topics: dict[str, list[list[str]]]) -> dict[str, dict[str, dict[str, list[list[str]]]]]:
    output: dict[str, dict[str, dict[str, list[list[str]]]]] = {corpus: {} for corpus in CORPORA}
    config_prefix = str(frame.config_sha256.iloc[0])[:12]
    for corpus in CORPORA:
        for model in ["s3_axial", "s3_angular", "s3_combined"]:
            _, _, variant = MODEL_META[model]
            output[corpus][variant] = {}
            for n_topics in TOPIC_COUNTS:
                key = f"{corpus}|{model}|42|{n_topics}|{config_prefix}"
                topics = raw_topics.get(key)
                if topics is None or len(topics) != n_topics or min(map(len, topics)) < 10:
                    raise RuntimeError(f"Missing or invalid topic terms for {key}.")
                output[corpus][variant][str(n_topics)] = topics
    return output


def main() -> None:
    if not AUDIT_PATH.exists() or "Status: **PASS**" not in AUDIT_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("The all-seed audit must pass before publishing an artifact.")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_sha = canonical_hash(config)
    frame = pd.read_csv(CSV_PATH)
    expected_rows = len(CORPORA) * len(SEEDS) * len(TOPIC_COUNTS) * len(MODELS)
    if len(frame) != expected_rows or not frame.status.eq("ok").all():
        raise RuntimeError(f"Expected {expected_rows} ok rows, received {len(frame)} rows.")
    if frame.config_sha256.nunique() != 1 or frame.config_sha256.iloc[0] != config_sha:
        raise RuntimeError("Config provenance does not match the locked config.")
    if set(frame.corpus) != set(CORPORA) or set(frame.seed) != set(SEEDS) or set(frame.n_topics) != set(TOPIC_COUNTS) or set(frame.model) != set(MODELS):
        raise RuntimeError("Result grid does not match the locked experimental matrix.")

    primary = frame.loc[frame.seed.eq(42)].sort_values(["corpus", "model", "n_topics"])
    raw_topics = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    payload = {
        "artifactVersion": "s3-cafebert-full-v1",
        "artifactScope": "full-corpus seed-42 primary tables plus mean ± sample SD across seeds 11,29,42,47; topic terms are seed-42 S³ outputs",
        "protocol": {
            "primaryMetric": "WEC-in",
            "secondaryMetrics": ["Topic diversity", "alphabetic term rate"],
            "appendixMetric": "C_NPMI",
            "externalWec": "N/A for Vietnamese; Google News Word2Vec is not language-compatible",
            "primarySeed": 42,
            "sensitivitySeeds": SEEDS,
            "topicCounts": TOPIC_COUNTS,
            "topTermsPerTopic": 10,
            "encoder": config["encoder"]["checkpoint"],
            "pooling": config["encoder"]["pooling"],
            "normalizeEmbeddings": config["encoder"]["normalize_embeddings"],
            "maxLength": config["encoder"]["max_length"],
            "batchSize": config["encoder"]["batch_size"],
            "implementation": "turftopic.SemanticSignalSeparation; feature_importance axial/angular/combined",
            "baselineProtocol": "LDA and NMF use CountVectorizer(min_df=10); BERTopic uses UMAP + KMeans and a separate min_df=2 topic-representation vectorizer",
            "timingContract": config["runtime_contract"],
            "configSha256": config_sha,
        },
        "corpora": corpus_meta(),
        "primarySeedRows": [row_payload(row) for _, row in primary.iterrows()],
        "topicTermsSeed42": topic_payload(frame, raw_topics),
        "multiSeedSummaryByTopics": summary_payload(frame),
        "audit": {
            "status": "pass",
            "actualRows": int(frame.shape[0]),
            "expectedRows": expected_rows,
            "duplicateRows": 0,
            "invalidMetricRows": 0,
            "auditReport": "FULL_MULTISEED_AUDIT.md",
            "resultsSha256": hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "bytes": OUT.stat().st_size, "primaryRows": len(payload["primarySeedRows"]), "summaryRows": len(payload["multiSeedSummaryByTopics"]), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
