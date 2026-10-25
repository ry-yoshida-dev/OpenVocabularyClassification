import math
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from open_vocabulary_classification import (
    ClassificationResult,
    ClassifierBackend,
    ClassifierSettings,
    Device,
    EmbeddingLayout,
    ImageFit,
    ImagePooling,
    ImageTiling,
    LogitCombination,
    OpenVocabularyClassifier,
    Prompt,
    ScoreActivation,
    TemplateCombination,
    TextQuery,
)
from open_vocabulary_classification.array_types import FloatArray


def unit_vector(angle: float) -> list[float]:
    return [math.cos(angle), math.sin(angle)]


class StubClassifier(OpenVocabularyClassifier):
    """Embeds an image at the angle width / 10 and a text at the angle len(text) / 10."""

    BACKEND: ClassVar[ClassifierBackend] = ClassifierBackend.CLIP

    def __init__(self, settings: ClassifierSettings) -> None:
        super().__init__(settings)
        self.mini_batch_sizes: list[int] = []
        self.image_modes: set[str] = set()
        self.images: list[Image.Image] = []
        self.embedded_texts: list[list[str]] = []

    @property
    def logit_scale(self) -> float:
        return 10.0

    @property
    def _letterbox_fill_color(self) -> tuple[int, int, int]:
        return (1, 2, 3)

    @property
    def logit_bias(self) -> float:
        return -1.0

    def _embed_mini_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        self.mini_batch_sizes.append(len(images))
        self.image_modes.update(image.mode for image in images)
        self.images.extend(images)
        return np.array([unit_vector(image.width / 10) for image in images], dtype=np.float64)

    def _embed_texts(self, texts: Sequence[str]) -> FloatArray:
        self.embedded_texts.append(list(texts))
        return np.array([unit_vector(len(text) / 10) for text in texts], dtype=np.float64)

    def _embed_layout(self, layout: EmbeddingLayout) -> FloatArray:
        sentence_embeddings: FloatArray = self._embed_texts(
            [sentence for entry in layout.entries for sentence in entry.sentences]
        )
        embeddings: list[FloatArray] = []
        start: int = 0
        for entry in layout.entries:
            mean: FloatArray = sentence_embeddings[start : start + len(entry.sentences)].mean(axis=0)
            embeddings.append(mean / np.linalg.norm(mean))
            start += len(entry.sentences)
        return np.stack(embeddings)


class BrokenClassifier(StubClassifier):
    def _embed_mini_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        return np.zeros((len(images) + 1, 2))

    def _embed_texts(self, texts: Sequence[str]) -> FloatArray:
        return np.zeros((len(texts) + 1, 2))

    def _embed_layout(self, layout: EmbeddingLayout) -> FloatArray:
        return np.zeros((len(layout) + 1, 2))


def build_settings(
    backend: ClassifierBackend = ClassifierBackend.CLIP,
    batch_size: int = 2,
    image_fit: ImageFit = ImageFit.LETTERBOX,
) -> ClassifierSettings:
    return ClassifierSettings(
        backend=backend,
        weights_path="stub",
        image_pooling=ImagePooling.PATCH_MEAN,
        image_fit=image_fit,
        batch_size=batch_size,
        device=Device.CPU,
    )


def build_sentence_settings(
    template_combination: TemplateCombination, query_combination: LogitCombination
) -> ClassifierSettings:
    return replace(
        build_settings(),
        text_templates=("{}", "a {}"),
        template_combination=template_combination,
        query_combination=query_combination,
    )


def expected_logit(image_width: int, text: str) -> float:
    return 10.0 * math.cos(image_width / 10 - len(text) / 10) - 1.0


IMAGES: list[Image.Image] = [Image.new("L", (width, 8)) for width in (10, 11, 12, 13, 14)]
PROMPT: Prompt = Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)})


def test_classify_images_batches_converts_and_scores() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    results: list[ClassificationResult] = classifier.classify_images(IMAGES, PROMPT)
    assert classifier.mini_batch_sizes == [2, 2, 1]
    assert classifier.image_modes == {"RGB"}
    assert len(results) == 5
    np.testing.assert_allclose(
        results[3].embedding_logits, [expected_logit(13, text) for text in ("dog", "puppy", "cat")]
    )
    assert results[3].score_activation is ScoreActivation.SOFTMAX
    assert results[3].top.matched_query == TextQuery("puppy")
    np.testing.assert_allclose(classifier.classify(IMAGES[0], PROMPT).embedding_logits, results[0].embedding_logits)


def test_classify_embeddings_equals_classify_images() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    embeddings: FloatArray = classifier.embed_images(IMAGES)
    image_tower_calls: int = len(classifier.mini_batch_sizes)
    from_embeddings: list[ClassificationResult] = classifier.classify_embeddings(embeddings, PROMPT)
    assert len(classifier.mini_batch_sizes) == image_tower_calls
    from_images: list[ClassificationResult] = classifier.classify_images(IMAGES, PROMPT)
    for embedded, direct in zip(from_embeddings, from_images, strict=True):
        np.testing.assert_allclose(embedded.embedding_logits, direct.embedding_logits)


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        (np.zeros((0, 2)), "shape"),
        (np.zeros((1, 0, 2)), "shape"),
        (np.ones(2), "shape"),
        (np.ones((1, 1, 1, 2)), "shape"),
        (np.array([[np.nan, 1.0]]), "finite"),
        (np.array([[2.0, 0.0]]), "L2-normalized"),
        (np.array([[1.0, 0.0, 0.0]]), "dimension 2"),
    ],
)
def test_classify_embeddings_validates_embeddings(embeddings: FloatArray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StubClassifier(build_settings()).classify_embeddings(embeddings, PROMPT)


def test_iter_classify_images_reads_one_mini_batch_at_a_time() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    loaded_count: list[int] = [0]

    def load_images() -> Iterator[Image.Image]:
        for image in IMAGES:
            loaded_count[0] += 1
            yield image

    results: Iterator[ClassificationResult] = classifier.iter_classify_images(load_images(), PROMPT)
    next(results)
    assert loaded_count[0] == 2
    assert len(list(results)) == 4
    assert list(classifier.iter_classify_images(iter([]), PROMPT)) == []


def test_tiling_keeps_the_best_logit_of_every_view() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    tiling: ImageTiling = ImageTiling(grid_size=2, overlap_ratio=0.0)
    result: ClassificationResult = classifier.classify(Image.new("RGB", (40, 8)), PROMPT, tiling)
    assert classifier.mini_batch_sizes == [2, 2, 1]
    assert sorted(image.width for image in classifier.images) == [20, 20, 20, 20, 40]
    np.testing.assert_allclose(
        result.embedding_logits,
        [max(expected_logit(width, text) for width in (20, 40)) for text in ("dog", "puppy", "cat")],
    )


def test_tiling_fills_mini_batches_with_the_views_of_several_images() -> None:
    classifier: StubClassifier = StubClassifier(build_settings(batch_size=10))
    tiling: ImageTiling = ImageTiling(grid_size=2)
    results: list[ClassificationResult] = classifier.classify_images(IMAGES, PROMPT, tiling)
    assert classifier.mini_batch_sizes == [10, 10, 5]
    for image, result in zip(IMAGES, results, strict=True):
        np.testing.assert_allclose(result.embedding_logits, classifier.classify(image, PROMPT, tiling).embedding_logits)


def test_classify_embeddings_equals_classify_images_with_tiling() -> None:
    classifier: StubClassifier = StubClassifier(build_settings(batch_size=10))
    tiling: ImageTiling = ImageTiling(grid_size=2)
    view_embeddings: FloatArray = classifier.embed_images(IMAGES, tiling)
    assert view_embeddings.shape == (5, tiling.view_count, 2)
    np.testing.assert_allclose(classifier.embed_image(IMAGES[2], tiling), view_embeddings[2])
    np.testing.assert_allclose(np.stack(list(classifier.iter_embed_images(iter(IMAGES), tiling))), view_embeddings)
    from_embeddings: list[ClassificationResult] = classifier.classify_embeddings(view_embeddings, PROMPT)
    from_images: list[ClassificationResult] = classifier.classify_images(IMAGES, PROMPT, tiling)
    for embedded, direct in zip(from_embeddings, from_images, strict=True):
        np.testing.assert_allclose(embedded.embedding_logits, direct.embedding_logits)


def test_embed_images_concatenates_mini_batches() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    embeddings: FloatArray = classifier.embed_images(IMAGES)
    assert embeddings.shape == (5, 2)
    np.testing.assert_allclose(embeddings[1], unit_vector(1.1))
    np.testing.assert_allclose(classifier.embed_image(IMAGES[1]), unit_vector(1.1))
    np.testing.assert_allclose(np.stack(list(classifier.iter_embed_images(iter(IMAGES)))), embeddings)
    assert classifier.mini_batch_sizes == [2, 2, 1, 1, 2, 2, 1]
    assert classifier.image_pooling is ImagePooling.PATCH_MEAN


def test_embed_texts_strips_and_scores_like_prompts() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    text_embeddings: FloatArray = classifier.embed_texts([" dog ", "puppy"])
    assert classifier.embedded_texts == [["dog", "puppy"]]
    image_embeddings: FloatArray = classifier.embed_images(IMAGES[:1])
    logits: FloatArray = image_embeddings @ text_embeddings.T * classifier.logit_scale + classifier.logit_bias
    np.testing.assert_allclose(logits[0], classifier.classify(IMAGES[0], PROMPT).embedding_logits[:2])
    np.testing.assert_allclose(classifier.embed_prompt(PROMPT)[:2], text_embeddings)


def test_empty_inputs_raise() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    with pytest.raises(ValueError, match="at least one image"):
        classifier.classify_images([], Prompt.from_class_names(("dog",)))
    with pytest.raises(ValueError, match="at least one image"):
        classifier.embed_images([])
    with pytest.raises(ValueError, match="at least one text"):
        classifier.embed_texts([])
    with pytest.raises(ValueError, match="blank"):
        classifier.embed_texts([" "])


def test_model_output_shapes_are_checked() -> None:
    classifier: BrokenClassifier = BrokenClassifier(build_settings())
    with pytest.raises(RuntimeError, match="embeddings of shape"):
        classifier.classify(IMAGES[0], Prompt.from_class_names(("dog",)))
    with pytest.raises(RuntimeError, match="embeddings of shape"):
        classifier.embed_image(IMAGES[0])
    with pytest.raises(RuntimeError, match="embeddings of shape"):
        classifier.embed_texts(["dog"])


def test_prompt_is_embedded_once_per_prompt_and_settings() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    for image in IMAGES:
        classifier.classify(image, PROMPT)
    classifier.classify_embeddings(classifier.embed_images(IMAGES), PROMPT)
    assert classifier.embedded_texts == [["dog", "puppy", "cat"]]
    assert not classifier.embed_prompt(PROMPT).flags.writeable
    classifier.settings = replace(build_settings(), text_templates=("a {}",))
    classifier.classify(IMAGES[0], PROMPT)
    assert classifier.embedded_texts[-1] == ["a dog", "a puppy", "a cat"]


def test_settings_of_another_backend_are_rejected() -> None:
    with pytest.raises(ValueError, match="needs clip settings"):
        StubClassifier(build_settings(ClassifierBackend.SIGLIP))


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGBA", (10, 8), (0, 0, 0, 0)),
        Image.new("LA", (10, 8), (0, 0)),
    ],
)
def test_transparent_pixels_become_white(image: Image.Image) -> None:
    classifier: StubClassifier = StubClassifier(build_settings(image_fit=ImageFit.NATIVE))
    classifier.embed_image(image)
    assert classifier.images[0].mode == "RGB"
    assert classifier.images[0].getpixel((0, 0)) == (255, 255, 255)


def test_palette_transparency_becomes_white() -> None:
    image: Image.Image = Image.new("P", (10, 8), 0)
    image.putpalette([0, 0, 0, 255, 0, 0])
    image.info["transparency"] = 0
    classifier: StubClassifier = StubClassifier(build_settings(image_fit=ImageFit.NATIVE))
    classifier.embed_image(image)
    assert classifier.images[0].getpixel((0, 0)) == (255, 255, 255)


def test_opaque_images_keep_their_colors() -> None:
    classifier: StubClassifier = StubClassifier(build_settings(image_fit=ImageFit.NATIVE))
    classifier.embed_image(Image.new("RGBA", (10, 8), (10, 20, 30, 255)))
    classifier.embed_image(Image.new("L", (10, 8), 40))
    assert [image.getpixel((0, 0)) for image in classifier.images] == [(10, 20, 30), (40, 40, 40)]


@pytest.mark.parametrize(
    ("image", "expected_values"),
    [
        (Image.fromarray(np.array([[1000, 2000], [3000, 5000]], dtype=np.uint16)), [0, 64, 128, 255]),
        (Image.fromarray(np.array([[-1.0, 0.0], [np.nan, 1.0]], dtype=np.float32)), [0, 128, 0, 255]),
        (Image.fromarray(np.full((2, 2), 7.0, dtype=np.float32)), [0, 0, 0, 0]),
    ],
)
def test_high_bit_depth_images_are_stretched(image: Image.Image, expected_values: list[int]) -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    classifier.embed_image(image)
    converted: Image.Image = classifier.images[0]
    assert converted.mode == "RGB"
    assert [converted.getpixel((x, y)) for y in range(2) for x in range(2)] == [
        (value, value, value) for value in expected_values
    ]


class AspectRatioKeepingClassifier(StubClassifier):
    IS_ASPECT_RATIO_KEPT: ClassVar[bool] = True


def test_letterbox_pads_images_and_tiles_to_squares() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    classifier.classify(Image.new("RGB", (4, 10), (200, 100, 50)), PROMPT, ImageTiling(grid_size=2, overlap_ratio=0.0))
    assert [image.size for image in classifier.images] == [(10, 10), (5, 5), (5, 5), (5, 5), (5, 5)]
    assert classifier.images[0].getpixel((0, 0)) == (1, 2, 3)
    assert classifier.images[0].getpixel((5, 5)) == (200, 100, 50)


@pytest.mark.parametrize(
    ("classifier_type", "image_fit"),
    [(StubClassifier, ImageFit.NATIVE), (AspectRatioKeepingClassifier, ImageFit.LETTERBOX)],
)
def test_images_keep_their_size_without_letterbox(classifier_type: type[StubClassifier], image_fit: ImageFit) -> None:
    classifier: StubClassifier = classifier_type(build_settings(image_fit=image_fit))
    classifier.embed_image(Image.new("RGB", (4, 10)))
    assert [image.size for image in classifier.images] == [(4, 10)]


def test_per_sentence_templates_are_combined_as_the_settings_say() -> None:
    sentences: dict[str, float] = {
        sentence: expected_logit(13, sentence) for sentence in ("dog", "a dog", "puppy", "a puppy")
    }
    classifier: StubClassifier = StubClassifier(build_sentence_settings(TemplateCombination.MEAN, LogitCombination.MIN))
    result: ClassificationResult = classifier.classify(IMAGES[3], PROMPT)
    assert len(classifier.embedding_layout(PROMPT)) == 6
    dog_mean: float = (sentences["dog"] + sentences["a dog"]) / 2
    puppy_mean: float = (sentences["puppy"] + sentences["a puppy"]) / 2
    assert result.classification(0).logit == pytest.approx(min(dog_mean, puppy_mean))
    classifier.settings = build_sentence_settings(TemplateCombination.SEPARATE, LogitCombination.MEAN)
    assert classifier.classify(IMAGES[3], PROMPT).classification(0).logit == pytest.approx(sum(sentences.values()) / 4)
    best_sentence: str = max(sentences, key=lambda sentence: sentences[sentence])
    assert classifier.classify(IMAGES[3], PROMPT).classification(0).matched_template == (
        "{}" if best_sentence in ("dog", "puppy") else "a {}"
    )


def test_ensembled_templates_report_no_template() -> None:
    classifier: StubClassifier = StubClassifier(replace(build_settings(), text_templates=("{}", "a {}")))
    result: ClassificationResult = classifier.classify(IMAGES[0], PROMPT)
    assert len(classifier.embedding_layout(PROMPT)) == 3
    assert result.classification(0).matched_template is None
