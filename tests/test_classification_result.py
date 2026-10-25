import json
import math
from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from open_vocabulary_classification import (
    AggregationMethod,
    AggregationStep,
    AggregationTiming,
    AggregationUnit,
    Classification,
    ClassificationRecord,
    ClassificationResult,
    ClassScores,
    EmbeddingEntry,
    EmbeddingLayout,
    EntryLabel,
    LogitAggregation,
    Prompt,
    PromptAggregation,
    ScoreActivation,
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


def aggregation(*specifications: str) -> PromptAggregation:
    steps: list[AggregationStep] = []
    for specification in specifications:
        unit, timing, method = specification.split(":")
        steps.append(AggregationStep(AggregationUnit(unit), AggregationTiming(timing), AggregationMethod(method)))
    return PromptAggregation.of(steps)


SIMILARITY_MAX: PromptAggregation = aggregation("template:similarity:max", "prompt:similarity:max")


def build_layout(
    prompt: Prompt = PROMPT,
    templates: TextTemplates = PLAIN_TEMPLATES,
    prompt_aggregation: PromptAggregation = SIMILARITY_MAX,
) -> EmbeddingLayout:
    return EmbeddingLayout.of(prompt, templates, prompt_aggregation)


def scored(
    layout: EmbeddingLayout, *, embedding_logits: FloatArray, score_activation: ScoreActivation
) -> ClassificationResult:
    return ClassificationResult(
        layout=layout,
        aggregation=LogitAggregation.of(layout),
        embedding_logits=embedding_logits,
        score_activation=score_activation,
    )


def build_result(score_activation: ScoreActivation, query_logits: FloatArray = QUERY_LOGITS) -> ClassificationResult:
    return scored(
        build_layout(),
        embedding_logits=query_logits,
        score_activation=score_activation,
    )


def build_sentence_result(prompt_aggregation: PromptAggregation) -> ClassificationResult:
    return scored(
        build_layout(DOG_PROMPT, PHOTO_TEMPLATES, prompt_aggregation),
        embedding_logits=SENTENCE_LOGITS,
        score_activation=ScoreActivation.SIGMOID,
    )


def test_each_class_takes_its_best_query() -> None:
    result: ClassificationResult = build_result(ScoreActivation.SOFTMAX)
    assert result.class_logits.tolist() == [3.0, 2.0, -1.0]
    assert result.matched_entry_ids.tolist() == [1, 2, 3]
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
    result: ClassificationResult = scored(
        EmbeddingLayout.of(prompt, PLAIN_TEMPLATES, PromptAggregation()),
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
    ("steps", "dog_logit"),
    [
        (("template:similarity:max", "prompt:similarity:max"), 3.0),
        (("template:similarity:mean", "prompt:similarity:max"), 2.5),
        (("template:similarity:min", "prompt:similarity:max"), 2.0),
        (("template:similarity:mean", "prompt:similarity:mean"), 2.125),
        (("template:similarity:max", "prompt:similarity:min"), 2.5),
        (("template:similarity:min", "prompt:similarity:min"), 1.0),
        (("prompt:similarity:max", "template:similarity:mean"), 2.5),
        (("template_and_prompt:similarity:mean",), 2.125),
        (("template_and_prompt:similarity:median",), 2.25),
        (("template_and_prompt:similarity:min",), 1.0),
        (
            ("template_and_prompt:similarity:logsumexp",),
            math.log(math.exp(3.0) + math.exp(2.0) + math.exp(2.5) + math.exp(1.0)),
        ),
    ],
)
def test_similarity_steps_reduce_the_table_of_a_class(steps: tuple[str, ...], dog_logit: float) -> None:
    result: ClassificationResult = build_sentence_result(aggregation(*steps))
    dog: Classification = result.classification(0)
    assert dog.logit == pytest.approx(dog_logit)
    assert dog.matched_query == TextQuery("dog")
    assert dog.matched_template == "a photo of a {}"
    assert result.classification(1).matched_template == "a drawing of a {}"


def test_matched_entry_follows_the_aggregated_logit() -> None:
    logits: FloatArray = np.array([4.0, -4.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    result: ClassificationResult = scored(
        build_layout(DOG_PROMPT, PHOTO_TEMPLATES, aggregation("template:similarity:mean", "prompt:similarity:max")),
        embedding_logits=logits,
        score_activation=ScoreActivation.SIGMOID,
    )
    assert result.classification(0).matched_query == TextQuery("puppy")
    assert result.classification(0).logit == pytest.approx(1.0)
    assert result.matched_entry_ids.tolist()[0] == 2


def test_embedding_steps_merge_the_entries_of_the_layout() -> None:
    ensembled: EmbeddingLayout = build_layout(DOG_PROMPT, PHOTO_TEMPLATES, PromptAggregation())
    assert [(entry.class_id, entry.query_id, entry.template) for entry in ensembled.entries] == [
        (0, 0, None),
        (0, 1, None),
        (1, 2, None),
    ]
    assert ensembled.entries[1].sentences == ("a photo of a puppy", "a drawing of a puppy")
    by_template: EmbeddingLayout = build_layout(
        DOG_PROMPT, PHOTO_TEMPLATES, aggregation("prompt:embedding:mean", "template:similarity:max")
    )
    assert [(entry.class_id, entry.query_id, entry.template) for entry in by_template.entries] == [
        (0, None, "a photo of a {}"),
        (0, None, "a drawing of a {}"),
        (1, None, "a photo of a {}"),
        (1, None, "a drawing of a {}"),
    ]
    assert by_template.entries[0].sentences == ("a photo of a dog", "a photo of a puppy")
    representative: EmbeddingLayout = build_layout(
        DOG_PROMPT, PHOTO_TEMPLATES, aggregation("template_and_prompt:embedding:mean")
    )
    assert [len(entry.sentences) for entry in representative.entries] == [4, 2]


def test_merged_prompts_report_no_matched_query() -> None:
    result: ClassificationResult = scored(
        build_layout(DOG_PROMPT, PHOTO_TEMPLATES, aggregation("prompt:embedding:mean", "template:similarity:max")),
        embedding_logits=np.array([1.0, 2.0, 0.0, -1.0]),
        score_activation=ScoreActivation.SIGMOID,
    )
    dog: Classification = result.classification(0)
    assert dog.logit == 2.0
    assert dog.matched_query is None
    assert dog.matched_template == "a drawing of a {}"
    assert dog.to_record()["matched_query_kind"] is None


def test_visual_queries_form_a_group_of_their_own_per_template() -> None:
    reference: VisualReference = VisualReference(Image.new("RGB", (4, 4)))
    prompt: Prompt = Prompt({"dog": (TextQuery("dog"), VisualQuery((reference,)))})
    layout: EmbeddingLayout = build_layout(
        prompt, PHOTO_TEMPLATES, aggregation("prompt:embedding:mean", "template:similarity:max")
    )
    assert [entry.template for entry in layout.entries] == ["a photo of a {}", "a drawing of a {}", None]
    assert layout.entries[2] == EmbeddingEntry(0, None, None, (), (1,))
    result: ClassificationResult = scored(
        layout,
        embedding_logits=np.array([0.0, 1.0, 5.0]),
        score_activation=ScoreActivation.SIGMOID,
    )
    assert result.classification(0).logit == 5.0
    assert result.classification(0).matched_template is None


def test_without_steps_every_sentence_is_a_candidate_and_a_class_takes_its_best() -> None:
    logits: FloatArray = np.array([3.0, 2.0, 2.5, 1.0, 0.0, 4.0])
    result: ClassificationResult = scored(
        build_layout(DOG_PROMPT, PHOTO_TEMPLATES, PromptAggregation.of([])),
        embedding_logits=logits,
        score_activation=ScoreActivation.SOFTMAX,
    )
    sentence_scores: FloatArray = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(result.scores, [sentence_scores[0], sentence_scores[5]])
    assert result.class_logits.tolist() == [3.0, 4.0]
    assert result.matched_entry_ids.tolist() == [0, 5]
    assert result.top.class_name == "cat"
    assert result.classification(0).matched_template == "a photo of a {}"


def test_a_single_axis_leaves_one_candidate_per_remaining_row_or_column() -> None:
    result: ClassificationResult = scored(
        build_layout(DOG_PROMPT, PHOTO_TEMPLATES, aggregation("template:similarity:mean")),
        embedding_logits=np.array([3.0, 2.0, 2.5, 1.0, 0.0, 4.0]),
        score_activation=ScoreActivation.SIGMOID,
    )
    assert result.class_logits.tolist() == [2.5, 2.0]
    assert result.classification(0).matched_query == TextQuery("dog")


def test_embedding_steps_must_not_follow_similarity_steps() -> None:
    with pytest.raises(ValueError, match="must not follow similarity"):
        aggregation("template:similarity:max", "prompt:embedding:mean")
    layout: EmbeddingLayout = build_layout(
        DOG_PROMPT, PHOTO_TEMPLATES, aggregation("prompt:embedding:mean", "template:similarity:max")
    )
    assert [(entry.class_id, entry.query_id, entry.template) for entry in layout.entries] == [
        (0, None, "a photo of a {}"),
        (0, None, "a drawing of a {}"),
        (1, None, "a photo of a {}"),
        (1, None, "a drawing of a {}"),
    ]
    result: ClassificationResult = scored(
        layout,
        embedding_logits=np.array([1.0, 2.0, 0.0, -1.0]),
        score_activation=ScoreActivation.SIGMOID,
    )
    assert result.class_logits.tolist() == [2.0, 0.0]


def test_stacked_embedding_steps_keep_the_groups_of_every_step() -> None:
    layout: EmbeddingLayout = build_layout(
        DOG_PROMPT, PHOTO_TEMPLATES, aggregation("template:embedding:mean", "prompt:embedding:mean")
    )
    assert layout.embedding_group_ids == ((0, 0, 1, 1, 2, 2), (0, 0, 1))
    assert [len(entry.sentences) for entry in layout.entries] == [4, 2]


def test_steps_embed_by_mean_only_and_read_from_plain_values() -> None:
    with pytest.raises(ValueError, match="by mean only"):
        AggregationStep(AggregationUnit.TEMPLATE, AggregationTiming.EMBEDDING, AggregationMethod.MAX)
    step: AggregationStep = AggregationStep.from_mapping({"unit": "prompt", "timing": "similarity", "method": "median"})
    assert AggregationStep.from_mapping(step.to_mapping()) == step
    with pytest.raises(KeyError, match="exactly the keys"):
        AggregationStep.from_mapping({"unit": "prompt", "timing": "similarity"})


@pytest.mark.parametrize("method", list(AggregationMethod))
def test_segments_match_combining_each_segment_on_its_own(method: AggregationMethod) -> None:
    labels: list[int] = [0, 0, 0, 1, 2, 2, 2, 2, 3, 3]
    logits: FloatArray = np.random.default_rng(0).normal(size=len(labels))
    logits[5] = logits[7] = logits[4:8].max() + 1.0
    segments: LogitSegments = LogitSegments.of(np.array(labels, dtype=np.int64))
    reductions: dict[AggregationMethod, Callable[[FloatArray], float]] = {
        AggregationMethod.MAX: lambda group: float(group.max()),
        AggregationMethod.MEAN: lambda group: float(group.mean()),
        AggregationMethod.MEDIAN: lambda group: float(np.median(group)),
        AggregationMethod.MIN: lambda group: float(group.min()),
        AggregationMethod.LOGSUMEXP: lambda group: float(np.log(np.exp(group).sum())),
    }
    groups: list[FloatArray] = [logits[np.array(labels) == label] for label in range(4)]
    np.testing.assert_allclose(segments.combine(logits, method), [reductions[method](group) for group in groups])
    assert segments.argmax(logits).tolist() == [int(np.argmax(logits[0:3])), 3, 5, 8 + int(np.argmax(logits[8:]))]


def test_logsumexp_then_softmax_sums_the_softmax_of_every_logit() -> None:
    logits: FloatArray = np.array([3.0, 2.0, 2.5, 1.0, 0.0, 4.0])
    result: ClassificationResult = scored(
        build_layout(DOG_PROMPT, PHOTO_TEMPLATES, aggregation("template_and_prompt:similarity:logsumexp")),
        embedding_logits=logits,
        score_activation=ScoreActivation.SOFTMAX,
    )
    sentence_scores: FloatArray = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(result.scores, [sentence_scores[:4].sum(), sentence_scores[4:].sum()])


def test_singleton_segments_return_the_logits() -> None:
    segments: LogitSegments = LogitSegments.of(np.array([0, 1, 2], dtype=np.int64))
    logits: FloatArray = np.array([2.0, -1.0, 0.5])
    assert segments.is_singleton
    assert segments.combine(logits, AggregationMethod.MEDIAN).tolist() == [2.0, -1.0, 0.5]
    assert segments.argmax(logits).tolist() == [0, 1, 2]


def test_invalid_segments_raise() -> None:
    with pytest.raises(ValueError, match="starts must increase"):
        LogitSegments(starts=np.array([0, 2, 2], dtype=np.int64), size=4)
    with pytest.raises(ValueError, match="starts must increase"):
        LogitSegments(starts=np.array([1], dtype=np.int64), size=4)
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        LogitSegments(starts=np.array([0, 2], dtype=np.int64), size=4).combine(np.zeros(3), AggregationMethod.MAX)


def test_layouts_compare_and_hash_by_their_fields() -> None:
    assert build_layout(DOG_PROMPT, PHOTO_TEMPLATES) == build_layout(DOG_PROMPT, PHOTO_TEMPLATES)
    assert hash(build_layout(DOG_PROMPT, PHOTO_TEMPLATES)) == hash(build_layout(DOG_PROMPT, PHOTO_TEMPLATES))
    assert build_layout(DOG_PROMPT, PHOTO_TEMPLATES) != build_layout(DOG_PROMPT)


def test_stored_logits_are_aggregated_again_from_their_labels() -> None:
    layout: EmbeddingLayout = build_layout(DOG_PROMPT, PHOTO_TEMPLATES, PromptAggregation.of([]))
    stored_labels: tuple[EntryLabel, ...] = tuple(
        EntryLabel(label.class_id, label.query_id, label.template) for label in layout.entry_labels
    )
    median: PromptAggregation = aggregation("template_and_prompt:similarity:median")
    again: ClassScores = ClassScores.of(
        LogitAggregation(stored_labels, median), SENTENCE_LOGITS, ScoreActivation.SIGMOID
    )
    expected: ClassificationResult = build_sentence_result(median)
    np.testing.assert_allclose(again.scores, expected.scores)
    assert again.class_logits.tolist() == expected.class_logits.tolist() == [2.25, 2.0]
    assert again.matched_entry_ids.tolist() == [0, 5]
    assert not again.scores.flags.writeable


def test_a_result_needs_the_aggregation_of_its_layout() -> None:
    layout: EmbeddingLayout = build_layout(DOG_PROMPT, PHOTO_TEMPLATES, SIMILARITY_MAX)
    with pytest.raises(ValueError, match="entry labels of the layout"):
        ClassificationResult(
            layout=layout,
            aggregation=LogitAggregation.of(build_layout()),
            embedding_logits=SENTENCE_LOGITS,
            score_activation=ScoreActivation.SIGMOID,
        )


def test_labels_merge_along_the_axis_of_a_unit() -> None:
    label: EntryLabel = EntryLabel(1, 3, "a photo of a {}")
    assert label.merged(AggregationUnit.TEMPLATE) == EntryLabel(1, 3, None)
    assert label.merged(AggregationUnit.PROMPT) == EntryLabel(1, None, "a photo of a {}")
    assert label.merged(AggregationUnit.TEMPLATE_AND_PROMPT) == EntryLabel(1, None, None)
    with pytest.raises(ValueError, match="ordered by class"):
        LogitAggregation((EntryLabel(1, 0, None), EntryLabel(0, 1, None)), SIMILARITY_MAX)
