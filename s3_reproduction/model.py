from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import FastICA


@dataclass
class S3TopicModel:
    n_topics: int = 20
    random_state: int = 42
    max_iter: int = 1000
    tol: float = 1e-4

    def fit(self, document_embeddings: np.ndarray) -> "S3TopicModel":
        if self.n_topics > min(document_embeddings.shape):
            raise ValueError("n_topics phải nhỏ hơn cả số tài liệu và kích thước embedding")
        self.ica_ = FastICA(
            n_components=self.n_topics,
            whiten="unit-variance",
            algorithm="parallel",
            whiten_solver="svd",
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.document_topic_ = self.ica_.fit_transform(document_embeddings)
        # sklearn components_ is the unmixing matrix C in the paper.
        self.unmixing_ = self.ica_.components_
        return self

    def transform(self, document_embeddings: np.ndarray) -> np.ndarray:
        return self.ica_.transform(document_embeddings)

    def word_scores(self, word_embeddings: np.ndarray, method: str = "combined") -> np.ndarray:
        # FastICA centers document embeddings before whitening. Vocabulary must
        # enter the same coordinate system before projection onto its axes.
        projected = (word_embeddings - self.ica_.mean_) @ self.unmixing_.T
        if method == "axial":
            return projected.T
        norm = np.linalg.norm(projected, axis=1, keepdims=True).clip(min=1e-12)
        if method == "angular":
            return (projected / norm).T
        if method == "combined":
            return (projected**3 / norm).T
        raise ValueError(f"Phương pháp word importance không hợp lệ: {method}")

    @staticmethod
    def top_words(
        scores: np.ndarray, vocabulary: list[str], top_n: int = 10, positive: bool = True
    ) -> list[list[str]]:
        # Paper §3.1: axes have two poles; the lowest-scoring words give a
        # "negative definition" of the topic, not just noise to discard.
        words = np.asarray(vocabulary)
        if positive:
            return [words[np.argsort(row)[-top_n:][::-1]].tolist() for row in scores]
        return [words[np.argsort(row)[:top_n]].tolist() for row in scores]
