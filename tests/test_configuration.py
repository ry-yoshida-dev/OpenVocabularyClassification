from dataclasses import fields
from pathlib import Path

import pytest
import yaml

import open_vocabulary_classification
from open_vocabulary_classification import (
    ClassifierBackend,
    ClassifierSettings,
    Device,
    ImagePooling,
    ScoreActivation,
    TextTemplates,
)

PRESET_DIRECTORY: Path = Path(open_vocabulary_classification.__file__).parent / "config"


def load_preset_section(path: Path) -> dict[str, object]:
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    section: object = document["classifier"]
    assert isinstance(section, dict)
    return {str(key): value for key, value in section.items()}


def test_every_backend_has_presets() -> None:
    preset_backends: set[str] = {path.parent.name for path in PRESET_DIRECTORY.glob("*/*.yaml")}
    assert preset_backends == {backend.value for backend in ClassifierBackend}


@pytest.mark.parametrize("path", sorted(PRESET_DIRECTORY.glob("*/*.yaml")), ids=lambda path: path.stem)
def test_preset_builds_settings(path: Path) -> None:
    section: dict[str, object] = load_preset_section(path)
    assert set(section) == {field.name for field in fields(ClassifierSettings)}
    assert section["backend"] == path.parent.name
    text_templates: object = section["text_templates"]
    assert isinstance(text_templates, list)
    settings: ClassifierSettings = ClassifierSettings(
        backend=ClassifierBackend(str(section["backend"])),
        weights_path=str(section["weights_path"]),
        image_pooling=ImagePooling(str(section["image_pooling"])),
        text_templates=tuple(str(template) for template in text_templates),
        batch_size=int(str(section["batch_size"])),
        device=Device(str(section["device"])),
        is_half_precision_enabled=bool(section["is_half_precision_enabled"]),
    )
    assert settings.image_pooling in settings.backend.supported_image_poolings


def test_backend_capabilities() -> None:
    assert ClassifierBackend.CLIP.supported_image_poolings == frozenset(
        {ImagePooling.CLASS_TOKEN, ImagePooling.PATCH_MEAN}
    )
    assert ClassifierBackend.SIGLIP.supported_image_poolings == frozenset(
        {ImagePooling.ATTENTION, ImagePooling.PATCH_MEAN}
    )
    assert ClassifierBackend.CLIP.score_activation is ScoreActivation.SOFTMAX
    assert ClassifierBackend.SIGLIP.score_activation is ScoreActivation.SIGMOID


def test_invalid_settings_raise() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ClassifierSettings(
            backend=ClassifierBackend.CLIP, weights_path="clip", image_pooling=ImagePooling.CLASS_TOKEN, batch_size=0
        )
    with pytest.raises(ValueError, match="weights_path"):
        ClassifierSettings(backend=ClassifierBackend.CLIP, weights_path=" ", image_pooling=ImagePooling.CLASS_TOKEN)
    with pytest.raises(ValueError, match="does not offer class_token pooling"):
        ClassifierSettings(
            backend=ClassifierBackend.SIGLIP, weights_path="siglip", image_pooling=ImagePooling.CLASS_TOKEN
        )
    with pytest.raises(ValueError, match="does not offer attention pooling"):
        ClassifierSettings(backend=ClassifierBackend.CLIP, weights_path="clip", image_pooling=ImagePooling.ATTENTION)
    with pytest.raises(ValueError, match="exactly one bare"):
        ClassifierSettings(
            backend=ClassifierBackend.CLIP,
            weights_path="clip",
            image_pooling=ImagePooling.PATCH_MEAN,
            text_templates=("a photo",),
        )


def test_default_template_uses_the_bare_query() -> None:
    settings: ClassifierSettings = ClassifierSettings(
        backend=ClassifierBackend.SIGLIP, weights_path="siglip", image_pooling=ImagePooling.ATTENTION
    )
    assert TextTemplates(settings.text_templates).fill("dog") == ("dog",)
