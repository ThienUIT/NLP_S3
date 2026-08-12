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
) -> tuple[SemanticSignalSeparation, list[list[str]]]:
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
    topics = [
        [str(word) for word, _score in terms]
        for _topic_id, terms in model.get_topics(top_k=10)
    ]
    return model, topics

