from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, BatchEncoding, CLIPModel, ProcessorMixin, Siglip2Model, SiglipModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from open_vocabulary_classification import (
    ClassificationResult,
    ClassifierBackend,
    ClassifierSettings,
    Device,
    OpenVocabularyClassifier,
    Prompt,
)
from open_vocabulary_classification.array_types import FloatArray

pytestmark: pytest.MarkDecorator = pytest.mark.pretrained

CHECKPOINTS: dict[str, ClassifierBackend] = {
    "openai/clip-vit-base-patch32": ClassifierBackend.CLIP,
    "google/siglip-base-patch16-224": ClassifierBackend.SIGLIP,
    "google/siglip2-base-patch16-224": ClassifierBackend.SIGLIP,
    "google/siglip2-base-patch16-naflex": ClassifierBackend.SIGLIP,
}
RANDOM: np.random.Generator = np.random.default_rng(0)
IMAGES: list[Image.Image] = [
    Image.fromarray(RANDOM.integers(0, 256, (240, 320, 3), dtype=np.uint8)),
    Image.fromarray(RANDOM.integers(0, 256, (300, 200, 3), dtype=np.uint8)),
]
CLASS_NAMES: tuple[str, ...] = ("a photo of a cat", "a photo of a dog", "a red square")
COLOR_IMAGES: dict[str, Image.Image] = {
    color: Image.new("RGB", (224, 224), value)
    for color, value in {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}.items()
}


@pytest.fixture(scope="module", params=list(CHECKPOINTS))
def weights_path(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def classifier(weights_path: str) -> OpenVocabularyClassifier:
    backend: ClassifierBackend = CHECKPOINTS[weights_path]
    return ClassifierSettings(
        backend=backend, weights_path=weights_path, image_pooling=backend.trained_image_pooling, device=Device.CPU
    ).build()


type DualEncoderModel = CLIPModel | SiglipModel | Siglip2Model


def load_reference_model(weights_path: str) -> tuple[DualEncoderModel, ProcessorMixin]:
    model: object = AutoModel.from_pretrained(weights_path)
    processor: object = AutoProcessor.from_pretrained(weights_path)
    assert isinstance(model, CLIPModel | SiglipModel | Siglip2Model)
    assert isinstance(processor, ProcessorMixin)
    return model, processor


def reference_logits(weights_path: str) -> FloatArray:
    model, processor = load_reference_model(weights_path)
    padding: str = "max_length" if CHECKPOINTS[weights_path] is ClassifierBackend.SIGLIP else "longest"
    inputs: BatchEncoding = processor(text=list(CLASS_NAMES), images=IMAGES, padding=padding, return_tensors="pt")
    with torch.inference_mode():
        logits: torch.Tensor = cast(torch.Tensor, model(**inputs).logits_per_image)
    return logits.float().numpy().astype(np.float64)


def reference_image_embeddings(weights_path: str) -> FloatArray:
    model, processor = load_reference_model(weights_path)
    with torch.inference_mode():
        features: object = model.get_image_features(**processor(images=IMAGES, return_tensors="pt"))
    assert isinstance(features, BaseModelOutputWithPooling)
    assert features.pooler_output is not None
    return torch.nn.functional.normalize(features.pooler_output.float(), dim=-1).numpy().astype(np.float64)


def test_image_embeddings_match_transformers(classifier: OpenVocabularyClassifier, weights_path: str) -> None:
    np.testing.assert_allclose(classifier.embed_images(IMAGES), reference_image_embeddings(weights_path), atol=1e-4)


def test_logits_match_transformers(classifier: OpenVocabularyClassifier, weights_path: str) -> None:
    results: list[ClassificationResult] = classifier.classify_images(IMAGES, Prompt.from_class_names(CLASS_NAMES))
    np.testing.assert_allclose(
        np.stack([result.query_logits for result in results]), reference_logits(weights_path), atol=1e-3
    )


def test_solid_colors_are_named(classifier: OpenVocabularyClassifier) -> None:
    prompt: Prompt = Prompt.from_class_names(tuple(f"a {color} image" for color in COLOR_IMAGES))
    results: list[ClassificationResult] = classifier.classify_images(list(COLOR_IMAGES.values()), prompt)
    assert [result.top.class_name for result in results] == [f"a {color} image" for color in COLOR_IMAGES]


def test_text_search_finds_the_described_image(classifier: OpenVocabularyClassifier) -> None:
    similarities: FloatArray = (
        classifier.embed_images(list(COLOR_IMAGES.values()))
        @ classifier.embed_texts([f"a {color} image" for color in COLOR_IMAGES]).T
    )
    assert np.argmax(similarities, axis=0).tolist() == list(range(len(COLOR_IMAGES)))
