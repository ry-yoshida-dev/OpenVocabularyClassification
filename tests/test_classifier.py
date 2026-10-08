from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from open_vocabulary_classification import (
    ClassificationResult,
    ClassifierBackend,
    ClassifierSettings,
    Device,
    ImagePooling,
    OpenVocabularyClassifier,
    Prompt,
    ScoreActivation,
    TextQuery,
)
from open_vocabulary_classification.array_types import FloatArray


class StubClassifier(OpenVocabularyClassifier):
    BACKEND: ClassVar[ClassifierBackend] = ClassifierBackend.CLIP

    def __init__(self, settings: ClassifierSettings) -> None:
        super().__init__(settings)
        self.mini_batch_sizes: list[int] = []
        self.image_modes: set[str] = set()
        self.images: list[Image.Image] = []

    def _embed_mini_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        self._record(images)
        return np.array([[float(image.width), 0.0] for image in images], dtype=np.float64)

    def _score_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> FloatArray:
        self._record(images)
        query_logits: FloatArray = np.arange(len(prompt.queries), dtype=np.float64)
        return np.stack([query_logits + image.width for image in images])

    def _record(self, images: Sequence[Image.Image]) -> None:
        self.mini_batch_sizes.append(len(images))
        self.image_modes.update(image.mode for image in images)
        self.images.extend(images)


class BrokenClassifier(StubClassifier):
    def _embed_mini_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        return np.zeros((len(images) + 1, 2))

    def _score_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> FloatArray:
        return np.zeros((len(images), len(prompt.queries) + 1))


def build_settings(backend: ClassifierBackend = ClassifierBackend.CLIP) -> ClassifierSettings:
    image_pooling: ImagePooling = ImagePooling.PATCH_MEAN
    return ClassifierSettings(
        backend=backend, weights_path="stub", image_pooling=image_pooling, batch_size=2, device=Device.CPU
    )


IMAGES: list[Image.Image] = [Image.new("L", (width, 8)) for width in (10, 11, 12, 13, 14)]


def test_classify_images_batches_converts_and_scores() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    prompt: Prompt = Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)})
    results: list[ClassificationResult] = classifier.classify_images(IMAGES, prompt)
    assert classifier.mini_batch_sizes == [2, 2, 1]
    assert classifier.image_modes == {"RGB"}
    assert len(results) == 5
    assert results[3].query_logits.tolist() == [13.0, 14.0, 15.0]
    assert results[3].class_logits.tolist() == [14.0, 15.0]
    assert results[3].score_activation is ScoreActivation.SOFTMAX
    assert results[3].top.class_name == "cat"
    assert results[0].classification(0).matched_query == TextQuery("puppy")
    assert classifier.classify(IMAGES[0], prompt).query_logits.tolist() == [10.0, 11.0, 12.0]


def test_embed_images_concatenates_mini_batches() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    embeddings: FloatArray = classifier.embed_images(IMAGES)
    assert embeddings.shape == (5, 2)
    assert embeddings[:, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert classifier.embed_image(IMAGES[1]).tolist() == [11.0, 0.0]
    assert classifier.mini_batch_sizes == [2, 2, 1, 1]
    assert classifier.image_pooling is ImagePooling.PATCH_MEAN


def test_empty_images_raise() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    with pytest.raises(ValueError, match="at least one image"):
        classifier.classify_images([], Prompt.from_class_names(("dog",)))
    with pytest.raises(ValueError, match="at least one image"):
        classifier.embed_images([])


def test_model_output_shapes_are_checked() -> None:
    classifier: BrokenClassifier = BrokenClassifier(build_settings())
    with pytest.raises(RuntimeError, match="query logits of shape"):
        classifier.classify(IMAGES[0], Prompt.from_class_names(("dog",)))
    with pytest.raises(RuntimeError, match="embeddings of shape"):
        classifier.embed_image(IMAGES[0])


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
    classifier: StubClassifier = StubClassifier(build_settings())
    classifier.embed_image(image)
    assert classifier.images[0].mode == "RGB"
    assert classifier.images[0].getpixel((0, 0)) == (255, 255, 255)


def test_palette_transparency_becomes_white() -> None:
    image: Image.Image = Image.new("P", (10, 8), 0)
    image.putpalette([0, 0, 0, 255, 0, 0])
    image.info["transparency"] = 0
    classifier: StubClassifier = StubClassifier(build_settings())
    classifier.embed_image(image)
    assert classifier.images[0].getpixel((0, 0)) == (255, 255, 255)


def test_opaque_images_keep_their_colors() -> None:
    classifier: StubClassifier = StubClassifier(build_settings())
    classifier.embed_image(Image.new("RGBA", (10, 8), (10, 20, 30, 255)))
    classifier.embed_image(Image.new("L", (10, 8), 40))
    assert [image.getpixel((0, 0)) for image in classifier.images] == [(10, 20, 30), (40, 40, 40)]
