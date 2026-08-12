import numpy as np

from s3_reproduction.metrics import topic_diversity
from s3_reproduction.model import S3TopicModel


def test_s3_shapes_and_topic_words():
    rng = np.random.default_rng(42)
    sources = rng.laplace(size=(200, 3))
    mixing = rng.normal(size=(3, 8))
    documents = sources @ mixing
    words = rng.normal(size=(12, 8))
    model = S3TopicModel(n_topics=3).fit(documents)
    scores = model.word_scores(words)
    topics = model.top_words(scores, [f"w{i}" for i in range(12)], top_n=4)
    assert model.document_topic_.shape == (200, 3)
    assert scores.shape == (3, 12)
    assert len(topics) == 3 and all(len(topic) == 4 for topic in topics)
    expected = (words - model.ica_.mean_) @ model.unmixing_.T
    expected = (expected**3 / np.linalg.norm(expected, axis=1, keepdims=True)).T
    np.testing.assert_allclose(scores, expected)


def test_topic_diversity():
    assert topic_diversity([["a", "b"], ["b", "c"]]) == 0.75
