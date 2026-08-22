from __future__ import annotations

import numpy as np


def topic_diversity(topics: list[list[str]]) -> float:
    terms = [term for topic in topics for term in topic]
    return len(set(terms)) / len(terms) if terms else 0.0


def embedding_coherence(topics: list[list[str]], vocabulary: list[str], embeddings: np.ndarray) -> float:
    lookup = {word: vector for word, vector in zip(vocabulary, embeddings)}
    values: list[float] = []
    for topic in topics:
        matrix = np.stack([lookup[word] for word in topic])
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
        similarities = matrix @ matrix.T
        upper = similarities[np.triu_indices(len(topic), k=1)]
        if len(upper):
            values.append(float(upper.mean()))
    return float(np.mean(values)) if values else 0.0

