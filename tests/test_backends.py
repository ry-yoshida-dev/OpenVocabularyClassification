from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    BatchEncoding,
    CLIPModel,
    MetaClip2Model,
    ProcessorMixin,
    Siglip2Model,
    SiglipModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from open_vocabulary_classification import (
    ClassificationResult,
    ClassifierBackend,
    ClassifierSettings,
    Device,
    ImagePooling,
    ImageTiling,
    OpenVocabularyClassifier,
    Precision,
    Prompt,
    VisualQuery,
    VisualReference,
)
from open_vocabulary_classification.array_types import FloatArray
from open_vocabulary_classification.backends.clip import ClipArchitecture, ClipClassifier, MetaClip2Classifier
from open_vocabulary_classification.backends.siglip import (
    Siglip2NaFlexClassifier,
    SiglipArchitecture,
    SiglipClassifier,
)

type DualEncoderModel = CLIPModel | MetaClip2Model | SiglipModel | Siglip2Model

RANDOM: np.random.Generator = np.random.default_rng(0)
IMAGES: list[Image.Image] = [
    Image.fromarray(RANDOM.integers(0, 256, (30, 40, 3), dtype=np.uint8)),
    Image.fromarray(RANDOM.integers(0, 256, (48, 16, 3), dtype=np.uint8)),
    Image.fromarray(RANDOM.integers(0, 256, (24, 24, 3), dtype=np.uint8)),
]
CLASS_NAMES: tuple[str, ...] = ("cat", "dog", "a cat")
BACKENDS: dict[str, ClassifierBackend] = {
    "clip": ClassifierBackend.CLIP,
    "metaclip_2": ClassifierBackend.CLIP,
    "siglip": ClassifierBackend.SIGLIP,
    "siglip2": ClassifierBackend.SIGLIP,
}
NATIVE_POOLINGS: dict[ClassifierBackend, ImagePooling] = {
    ClassifierBackend.CLIP: ImagePooling.CLASS_TOKEN,
    ClassifierBackend.SIGLIP: ImagePooling.ATTENTION,
}


def build_classifier(
    checkpoint_directory: Path, name: str, image_pooling: ImagePooling | None = None, batch_size: int = 8
) -> OpenVocabularyClassifier:
    backend: ClassifierBackend = BACKENDS[name]
    return ClassifierSettings(
        backend=backend,
        weights_path=str(checkpoint_directory / name),
        image_pooling=NATIVE_POOLINGS[backend] if image_pooling is None else image_pooling,
        batch_size=batch_size,
        device=Device.CPU,
    ).build()


def load_reference_model(checkpoint_directory: Path, name: str) -> tuple[DualEncoderModel, ProcessorMixin]:
    model: object = AutoModel.from_pretrained(checkpoint_directory / name)
    processor: object = AutoProcessor.from_pretrained(checkpoint_directory / name)
    assert isinstance(model, CLIPModel | MetaClip2Model | SiglipModel | Siglip2Model)
    assert isinstance(processor, ProcessorMixin)
    return model, processor


def preprocess(processor: ProcessorMixin, texts: list[str] | None = None) -> BatchEncoding:
    inputs: BatchEncoding = processor.image_processor(images=IMAGES, return_tensors="pt")
    if texts is not None:
        inputs.update(processor.tokenizer(texts, padding="max_length", max_length=12, return_tensors="pt"))
    return inputs


def normalized(tensor: torch.Tensor) -> FloatArray:
    return torch.nn.functional.normalize(tensor.float(), dim=-1).numpy().astype(np.float64)


def pooled_output(outputs: object) -> torch.Tensor:
    assert isinstance(outputs, BaseModelOutputWithPooling)
    assert outputs.pooler_output is not None
    return outputs.pooler_output


@pytest.mark.parametrize(
    ("name", "classifier_type"),
    [
        ("clip", ClipClassifier),
        ("metaclip_2", MetaClip2Classifier),
        ("siglip", SiglipClassifier),
        ("siglip2", Siglip2NaFlexClassifier),
    ],
)
def test_architecture_follows_model_type(
    checkpoint_directory: Path, name: str, classifier_type: type[OpenVocabularyClassifier]
) -> None:
    assert type(build_classifier(checkpoint_directory, name)) is classifier_type


def test_checkpoint_of_another_family_is_rejected(checkpoint_directory: Path) -> None:
    with pytest.raises(ValueError, match="not one of"):
        ClipArchitecture.of_checkpoint(str(checkpoint_directory / "siglip"))
    with pytest.raises(ValueError, match="not one of"):
        SiglipArchitecture.of_checkpoint(str(checkpoint_directory / "metaclip_2"))


@pytest.mark.parametrize("name", list(BACKENDS))
def test_native_pooling_matches_the_model_image_features(checkpoint_directory: Path, name: str) -> None:
    model, processor = load_reference_model(checkpoint_directory, name)
    with torch.inference_mode():
        expected: FloatArray = normalized(pooled_output(model.get_image_features(**preprocess(processor))))
    embeddings: FloatArray = build_classifier(checkpoint_directory, name).embed_images(IMAGES)
    np.testing.assert_allclose(embeddings, expected, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_query_logits_match_the_model_logits(checkpoint_directory: Path, name: str) -> None:
    model, processor = load_reference_model(checkpoint_directory, name)
    with torch.inference_mode():
        inputs: BatchEncoding = preprocess(processor, list(CLASS_NAMES))
        if BACKENDS[name] is ClassifierBackend.SIGLIP:
            inputs.pop("attention_mask", None)
        expected: torch.Tensor = cast(torch.Tensor, model(**inputs).logits_per_image)
    results: list[ClassificationResult] = build_classifier(checkpoint_directory, name).classify_images(
        IMAGES, Prompt.from_class_names(CLASS_NAMES)
    )
    np.testing.assert_allclose(np.stack([result.query_logits for result in results]), expected.numpy(), atol=1e-4)
    scores: FloatArray = results[0].scores
    match BACKENDS[name]:
        case ClassifierBackend.CLIP:
            assert scores.sum() == pytest.approx(1.0)
        case ClassifierBackend.SIGLIP:
            np.testing.assert_allclose(scores, 1.0 / (1.0 + np.exp(-results[0].query_logits)))


def test_clip_patch_mean_projects_the_mean_of_normalized_patch_tokens(checkpoint_directory: Path) -> None:
    model, processor = load_reference_model(checkpoint_directory, "clip")
    assert isinstance(model, CLIPModel)
    with torch.inference_mode():
        pixel_values: torch.Tensor = preprocess(processor)["pixel_values"]
        tokens: torch.Tensor | None = model.vision_model(pixel_values=pixel_values).last_hidden_state
        assert tokens is not None
        patch_mean: torch.Tensor = model.vision_model.post_layernorm(tokens[:, 1:]).mean(dim=1)
        expected: FloatArray = normalized(model.visual_projection(patch_mean))
    embeddings: FloatArray = build_classifier(checkpoint_directory, "clip", ImagePooling.PATCH_MEAN).embed_images(
        IMAGES
    )
    np.testing.assert_allclose(embeddings, expected, atol=1e-5)
    class_token_embeddings: FloatArray = build_classifier(checkpoint_directory, "clip").embed_images(IMAGES)
    assert not np.allclose(embeddings, class_token_embeddings, atol=1e-3)


@pytest.mark.parametrize("image_pooling", [ImagePooling.ATTENTION, ImagePooling.PATCH_MEAN])
def test_naflex_padding_does_not_change_embeddings(checkpoint_directory: Path, image_pooling: ImagePooling) -> None:
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, "siglip2", image_pooling)
    batched: FloatArray = classifier.embed_images(IMAGES)
    one_by_one: FloatArray = np.stack([classifier.embed_image(image) for image in IMAGES])
    np.testing.assert_allclose(batched, one_by_one, atol=1e-5)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_patch_mean_is_offered_by_every_backend(checkpoint_directory: Path, name: str) -> None:
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name, ImagePooling.PATCH_MEAN)
    native: FloatArray = build_classifier(checkpoint_directory, name).embed_images(IMAGES)
    embeddings: FloatArray = classifier.embed_images(IMAGES)
    assert embeddings.shape == native.shape
    assert not np.allclose(embeddings, native, atol=1e-3)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_visual_query_matches_its_own_reference(checkpoint_directory: Path, name: str) -> None:
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name, batch_size=2)
    prompt: Prompt = Prompt(
        {f"image {index}": (VisualQuery((VisualReference(image),)),) for index, image in enumerate(IMAGES)}
    )
    results: list[ClassificationResult] = classifier.classify_images(IMAGES, prompt)
    assert [result.top.class_id for result in results] == [0, 1, 2]
    self_logits: FloatArray = np.diagonal(np.stack([result.query_logits for result in results]))
    np.testing.assert_allclose(self_logits, self_logits[0], rtol=1e-5)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_text_queries_longer_than_the_text_tower_raise(checkpoint_directory: Path, name: str) -> None:
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name)
    with pytest.raises(ValueError, match="must fit in 12 tokens"):
        classifier.classify(IMAGES[0], Prompt.from_class_names((" ".join(["cat"] * 20),)))
    with pytest.raises(ValueError, match=r"must fit in 12 tokens. got \d+ tokens for '(cat ){19}cat'"):
        classifier.embed_texts([" ".join(["cat"] * 14), " ".join(["cat"] * 20), "dog"])


@pytest.mark.parametrize("name", list(BACKENDS))
def test_classify_embeddings_matches_classify_images(checkpoint_directory: Path, name: str) -> None:
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name, batch_size=2)
    prompt: Prompt = Prompt.from_class_names(CLASS_NAMES)
    from_embeddings: list[ClassificationResult] = classifier.classify_embeddings(
        classifier.embed_images(IMAGES), prompt
    )
    from_images: list[ClassificationResult] = classifier.classify_images(IMAGES, prompt)
    np.testing.assert_allclose(
        np.stack([result.query_logits for result in from_embeddings]),
        np.stack([result.query_logits for result in from_images]),
        atol=1e-5,
    )


@pytest.mark.parametrize("name", list(BACKENDS))
def test_embed_texts_matches_the_model_text_features(checkpoint_directory: Path, name: str) -> None:
    model, processor = load_reference_model(checkpoint_directory, name)
    with torch.inference_mode():
        tokens: BatchEncoding = processor.tokenizer(
            list(CLASS_NAMES), padding="max_length", max_length=12, return_tensors="pt"
        )
        if BACKENDS[name] is ClassifierBackend.SIGLIP:
            tokens.pop("attention_mask", None)
        expected: FloatArray = normalized(pooled_output(model.get_text_features(**tokens)))
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name)
    np.testing.assert_allclose(classifier.embed_texts(CLASS_NAMES), expected, atol=1e-5)
    np.testing.assert_allclose(classifier.embed_prompt(Prompt.from_class_names(CLASS_NAMES)), expected, atol=1e-5)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_tiling_never_lowers_the_whole_image_logits(checkpoint_directory: Path, name: str) -> None:
    classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name)
    prompt: Prompt = Prompt.from_class_names(CLASS_NAMES)
    whole: list[ClassificationResult] = classifier.classify_images(IMAGES, prompt)
    tiled: list[ClassificationResult] = classifier.classify_images(IMAGES, prompt, ImageTiling(grid_size=2))
    for whole_result, tiled_result in zip(whole, tiled, strict=True):
        assert np.all(tiled_result.query_logits >= whole_result.query_logits - 1e-5)


@pytest.mark.parametrize("name", list(BACKENDS))
def test_bfloat16_embeddings_stay_close_to_float32(checkpoint_directory: Path, name: str) -> None:
    backend: ClassifierBackend = BACKENDS[name]
    classifier: OpenVocabularyClassifier = ClassifierSettings(
        backend=backend,
        weights_path=str(checkpoint_directory / name),
        image_pooling=backend.trained_image_pooling,
        device=Device.CPU,
        precision=Precision.BFLOAT16,
    ).build()
    embeddings: FloatArray = classifier.embed_images(IMAGES)
    assert embeddings.dtype == np.float64
    float32_classifier: OpenVocabularyClassifier = build_classifier(checkpoint_directory, name)
    assert classifier.logit_scale == float32_classifier.logit_scale
    assert classifier.logit_bias == float32_classifier.logit_bias
    np.testing.assert_allclose(embeddings, build_classifier(checkpoint_directory, name).embed_images(IMAGES), atol=0.05)
