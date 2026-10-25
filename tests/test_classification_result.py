import json
from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from open_vocabulary_classification import (
    Classification,
    ClassificationRecord,
    ClassificationResult,
    EmbeddingLayout,
    LogitAggregation,
    LogitCombination,
    Prompt,
    ScoreActivation,
    TemplateCombination,
    TextQuery,
    TextTemplates,
    VisualQuery,
    VisualReference,
)
from open_vocabulary_classification.array_types import FloatArray
from open_vocabulary_classification.result import LogitSegments

PROMPT: Prompt = Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",), "bird": ("bird",)})
QUERY_LOGITS: FloatArray = np.array([1.0, 3.0, 2.0, -1.0], dtype=np.float64)
PLAIN_TEMPLATES: TextTemplates = TextTemplates(("{}",))
PHOTO_TEMPLATES: TextTemplates = TextTemplates(("a photo of a {}", "a drawing of a {}"))
DOG_PROMPT: Prompt = Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)})
SENTENCE_LOGITS: FloatArray = np.array([3.0, 2.0, 2.5, 1.0, 0.0, 4.0], dtype=np.float64)


def build_layout(
    prompt: Prompt = PROMPT,
    templates: TextTemplates = PLAIN_TEMPLATES,
    template_combination: TemplateCombination = TemplateCombination.MAX,
) -> EmbeddingLayout:
    return EmbeddingLayout.of(prompt, templates, template_combination)


def build_result(score_activation: ScoreActivation, query_logits: FloatArray = QUERY_LOGITS) -> ClassificationResult:
    return ClassificationResult(
        aggregation=LogitAggregation(build_layout(), TemplateCombination.MAX),
        embedding_logits=query_logits,
        score_activation=score_activation,
    )


def build_sentence_result(
    template_combination: TemplateCombination, query_combination: LogitCombination
) -> ClassificationResult:
    return ClassificationResult(
        aggregation=LogitAggregation(
            build_layout(DOG_PROMPT, PHOTO_TEMPLATES, template_combination),
            template_combination=template_combination,
            query_combination=query_combination,
        ),
        embedding_logits=SENTENCE_LOGITS,
        score_activation=ScoreActivation.SIGMOID,
    )


def test_each_class_takes_its_best_query() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX)
    assert result.class_logits.tolist() == [3.0, 2.0, -1.0]
    assert result.matched_query_ids.tolist() == [1, 2, 3]
    np.testing.assert_allclose(result.scores, np.exp([3.0, 2.0, -1.0]) / np.exp([3.0, 2.0, -1.0]).sum())
    assert result.top == Classification(
        class_id=0,
        class_name="dog",
        score=float(result.scores[0]),
        logit=3.0,
        matched_query=TextQuery("puppy"),
        matched_template="{}",
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
    assert result.embedding_logits.tolist() == [1.0, 3.0, 2.0, -1.0]
    arrays: tuple[np.ndarray[tuple[int], np.dtype[np.generic]], ...] = (
        result.embedding_logits,
        result.class_logits,
        result.matched_entry_ids,
        result.matched_query_ids,
        result.scores,
    )
    for array in arrays:
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


def test_filter_by_logit_is_independent_of_other_classes() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX)
    assert [classification.class_name for classification in result.filter_by_logit(2.0)] == ["dog", "cat"]
    assert result.filter_by_logit(5.0) == ()
    with pytest.raises(ValueError, match="finite"):
        result.filter_by_logit(float("nan"))


def test_records_are_json_serializable() -> None:
    references: tuple[VisualReference, ...] = tuple(VisualReference(Image.new("RGB", (4, 4))) for _ in range(2))
    prompt: Prompt = Prompt({"dog": (TextQuery("dog"),), "my dog": (VisualQuery(references),)})
    result: ClassificationResult = ClassificationResult(
        aggregation=LogitAggregation(
            EmbeddingLayout.of(prompt, PLAIN_TEMPLATES, TemplateCombination.ENSEMBLE), TemplateCombination.ENSEMBLE
        ),
        embedding_logits=np.array([1.0, 2.0]),
        score_activation=ScoreActivation.SIGMOID,
    )
    records: list[ClassificationRecord] = result.to_records()
    assert records[0]["score"] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))
    assert records[0] == ClassificationRecord(
        class_id=0,
        class_name="dog",
        score=float(result.scores[0]),
        logit=1.0,
        matched_query_kind="text",
        matched_query_text="dog",
        matched_query_reference_count=None,
        matched_template=None,
    )
    assert records[1]["matched_query_kind"] == "visual"
    assert records[1]["matched_query_text"] is None
    assert records[1]["matched_query_reference_count"] == 2
    assert json.loads(json.dumps(records)) == records


@pytest.mark.parametrize(
    ("template_combination", "query_combination", "dog_logit", "matched_query", "matched_template"),
    [
        (TemplateCombination.MAX, LogitCombination.MAX, 3.0, "dog", "a photo of a {}"),
        (TemplateCombination.MEAN, LogitCombination.MAX, 2.5, "dog", "a photo of a {}"),
        (TemplateCombination.MIN, LogitCombination.MAX, 2.0, "dog", "a photo of a {}"),
        (TemplateCombination.MEAN, LogitCombination.MEAN, 2.125, "dog", "a photo of a {}"),
        (TemplateCombination.MAX, LogitCombination.MIN, 2.5, "dog", "a photo of a {}"),
        (TemplateCombination.MIN, LogitCombination.MIN, 1.0, "dog", "a photo of a {}"),
        (TemplateCombination.SEPARATE, LogitCombination.MEAN, 2.125, "dog", "a photo of a {}"),
        (TemplateCombination.SEPARATE, LogitCombination.MEDIAN, 2.25, "dog", "a photo of a {}"),
        (TemplateCombination.SEPARATE, LogitCombination.MIN, 1.0, "dog", "a photo of a {}"),
    ],
)
def test_sentences_and_queries_are_combined_in_two_steps(
    template_combination: TemplateCombination,
    query_combination: LogitCombination,
    dog_logit: float,
    matched_query: str,
    matched_template: str,
) -> None:
    result: ClassificationResult = build_sentence_result(template_combination, query_combination)
    dog: Classification = result.classification(0)
    assert dog.logit == pytest.approx(dog_logit)
    assert dog.matched_query == TextQuery(matched_query)
    assert dog.matched_template == matched_template
    assert result.classification(1).matched_template == "a drawing of a {}"


def test_matched_query_follows_the_combined_logit() -> None:
    logits: FloatArray = np.array([4.0, -4.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    result: ClassificationResult = ClassificationResult(
        aggregation=LogitAggregation(build_layout(DOG_PROMPT, PHOTO_TEMPLATES), TemplateCombination.MEAN),
        embedding_logits=logits,
        score_activation=ScoreActivation.SIGMOID,
    )
    assert result.classification(0).matched_query == TextQuery("puppy")
    assert result.classification(0).logit == pytest.approx(1.0)
    assert result.matched_entry_ids.tolist()[0] == 2


@pytest.mark.parametrize("combination", list(LogitCombination))
def test_segments_match_combining_each_segment_on_its_own(combination: LogitCombination) -> None:
    labels: list[int] = [0, 0, 0, 1, 2, 2, 2, 2, 3, 3]
    logits: FloatArray = np.random.default_rng(0).normal(size=len(labels))
    logits[5] = logits[7] = logits[4:8].max() + 1.0
    segments: LogitSegments = LogitSegments.of(np.array(labels, dtype=np.int64))
    reductions: dict[LogitCombination, Callable[[FloatArray], float]] = {
        LogitCombination.MAX: lambda group: float(group.max()),
        LogitCombination.MEAN: lambda group: float(group.mean()),
        LogitCombination.MEDIAN: lambda group: float(np.median(group)),
        LogitCombination.MIN: lambda group: float(group.min()),
    }
    groups: list[FloatArray] = [logits[np.array(labels) == label] for label in range(4)]
    np.testing.assert_allclose(
        segments.combine(logits, combination), [reductions[combination](group) for group in groups]
    )
    assert segments.argmax(logits).tolist() == [int(np.argmax(logits[0:3])), 3, 5, 8 + int(np.argmax(logits[8:]))]


def test_singleton_segments_return_the_logits() -> None:
    segments: LogitSegments = LogitSegments.of(np.array([0, 1, 2], dtype=np.int64))
    logits: FloatArray = np.array([2.0, -1.0, 0.5])
    assert segments.is_singleton
    assert segments.combine(logits, LogitCombination.MEDIAN).tolist() == [2.0, -1.0, 0.5]
    assert segments.argmax(logits).tolist() == [0, 1, 2]


def test_invalid_segments_raise() -> None:
    with pytest.raises(ValueError, match="starts must increase"):
        LogitSegments(starts=np.array([0, 2, 2], dtype=np.int64), size=4)
    with pytest.raises(ValueError, match="starts must increase"):
        LogitSegments(starts=np.array([1], dtype=np.int64), size=4)
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        LogitSegments(starts=np.array([0, 2], dtype=np.int64), size=4).combine(np.zeros(3), LogitCombination.MAX)


def test_layouts_compare_and_hash_by_prompt_and_entries() -> None:
    assert build_layout(DOG_PROMPT, PHOTO_TEMPLATES) == build_layout(DOG_PROMPT, PHOTO_TEMPLATES)
    assert hash(build_layout(DOG_PROMPT, PHOTO_TEMPLATES)) == hash(build_layout(DOG_PROMPT, PHOTO_TEMPLATES))
    assert build_layout(DOG_PROMPT, PHOTO_TEMPLATES) != build_layout(DOG_PROMPT)


def test_aggregation_rejects_a_layout_of_another_template_combination() -> None:
    with pytest.raises(ValueError, match="layout does not match"):
        LogitAggregation(
            build_layout(DOG_PROMPT, PHOTO_TEMPLATES, TemplateCombination.MAX), TemplateCombination.ENSEMBLE
        )
    with pytest.raises(ValueError, match="layout does not match"):
        LogitAggregation(
            build_layout(DOG_PROMPT, PHOTO_TEMPLATES, TemplateCombination.ENSEMBLE), TemplateCombination.MAX
        )


@pytest.mark.parametrize("template_combination", list(TemplateCombination))
def test_template_combinations_reduce_sentence_logits_unless_kept_whole(
    template_combination: TemplateCombination,
) -> None:
    sentence_combination: LogitCombination | None = template_combination.logit_combination
    match template_combination:
        case TemplateCombination.ENSEMBLE | TemplateCombination.SEPARATE:
            assert sentence_combination is None
        case _:
            assert sentence_combination is LogitCombination(template_combination.value)
    assert template_combination.is_ensembled == (template_combination is TemplateCombination.ENSEMBLE)
