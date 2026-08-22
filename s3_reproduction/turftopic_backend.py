from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import SemanticSignalSeparation

from .encoder import CafeBERTEncoder


@dataclass
class CachedVocabularyEncoder:
    """Expose CafeBERT's encode API while reusing one vocabulary embedding matrix."""

    encoder: CafeBERTEncoder
    vocabulary: list[str]
    vocabulary_embeddings: np.ndarray

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        values = list(texts)
        if values == self.vocabulary:
            return self.vocabulary_embeddings
        return self.encoder.encode(values)


def fit_turftopic(
    documents: list[str],
    document_embeddings: np.ndarray,
    vocabulary: list[str],
    vocabulary_embeddings: np.ndarray,
    vectorizer: CountVectorizer,
    encoder: CafeBERTEncoder,
    n_topics: int,
    random_state: int,
    top_n: int = 10,
) -> tuple[SemanticSignalSeparation, list[list[str]], list[list[str]]]:
    cached_encoder = CachedVocabularyEncoder(encoder, vocabulary, vocabulary_embeddings)
    model = SemanticSignalSeparation(
        n_components=n_topics,
        encoder=cached_encoder,
        vectorizer=vectorizer,
        max_iter=1000,
        feature_importance="combined",
        random_state=random_state,
    )
    model.fit_transform(documents, embeddings=document_embeddings)
    # Paper §3.1: axes have two poles; the lowest-scoring words give a
    # "negative definition" of the topic, not just noise to discard.
    topics = model.get_top_words(top_k=top_n, positive=True)
    topics_negative = model.get_top_words(top_k=top_n, positive=False)
    return model, topics, topics_negative

