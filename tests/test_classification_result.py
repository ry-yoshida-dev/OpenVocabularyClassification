import numpy as np
import pytest

from open_vocabulary_classification import (
    Classification,
    ClassificationResult,
    Prompt,
    ScoreActivation,
    TextQuery,
)
from open_vocabulary_classification.array_types import FloatArray

PROMPT: Prompt = Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",), "bird": ("bird",)})
QUERY_LOGITS: FloatArray = np.array([1.0, 3.0, 2.0, -1.0], dtype=np.float64)


def build_result(score_activation: ScoreActivation, query_logits: FloatArray = QUERY_LOGITS) -> ClassificationResult:
    return ClassificationResult(prompt=PROMPT, query_logits=query_logits, score_activation=score_activation)


def test_each_class_takes_its_best_query() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX)
    assert result.class_logits.tolist() == [3.0, 2.0, -1.0]
    assert result.matched_query_ids.tolist() == [1, 2, 3]
    np.testing.assert_allclose(result.scores, np.exp([3.0, 2.0, -1.0]) / np.exp([3.0, 2.0, -1.0]).sum())
    assert result.top == Classification(
        class_id=0, class_name="dog", score=float(result.scores[0]), logit=3.0, matched_query=TextQuery("puppy")
    )
    assert [classification.class_name for classification in result] == ["dog", "cat", "bird"]
    assert result.class_names == ("dog", "cat", "bird")
    assert len(result) == 3


def test_ranking_and_lookup() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SIGMOID)
    assert [classification.class_name for classification in result.top_k(2)] == ["dog", "cat"]
    assert len(result.top_k(10)) == 3
    assert [classification.class_name for classification in result.filter_by_score(0.5)] == ["dog", "cat"]
    assert result.filter_by_score(1.0) == ()
    assert result.score_of(" cat ") == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    assert result.classification(2).matched_query == TextQuery("bird")


def test_ties_keep_class_order() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX, np.array([0.0, 0.0, 1.0, 1.0]))
    assert [classification.class_id for classification in result.top_k(3)] == [1, 2, 0]


def test_invalid_access_raises() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX)
    with pytest.raises(ValueError, match="k must be positive"):
        result.top_k(0)
    with pytest.raises(ValueError, match="threshold"):
        result.filter_by_score(1.5)
    with pytest.raises(KeyError, match="'fish' is not one of"):
        result.score_of("fish")
    with pytest.raises(IndexError, match="class_id"):
        result.classification(3)


def test_invalid_logits_raise() -> None:
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        build_result(ScoreActivation.SOFTMAX, np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        build_result(ScoreActivation.SOFTMAX, np.array([0.0, np.nan, 0.0, 0.0]))


def test_score_activations_are_stable_for_large_logits() -> None:
    logits: FloatArray = np.array([[1000.0, 0.0, -1000.0]])
    np.testing.assert_allclose(ScoreActivation.SOFTMAX.apply(logits), [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(ScoreActivation.SIGMOID.apply(logits), [[1.0, 0.5, 0.0]])
    np.testing.assert_allclose(ScoreActivation.SOFTMAX.apply(np.array([1.0, 1.0])), [0.5, 0.5])


def test_arrays_are_read_only_copies() -> None:
    query_logits: FloatArray = QUERY_LOGITS.copy()
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX, query_logits)
    query_logits[0] = 100.0
    assert result.query_logits.tolist() == [1.0, 3.0, 2.0, -1.0]
    for array in (result.query_logits, result.class_logits, result.matched_query_ids, result.scores):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
