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
    Precision,
    PresetCatalog,
    ScoreActivation,
    TextTemplates,
)

PRESET_DIRECTORY: Path = Path(open_vocabulary_classification.__file__).parent / "config"
PRESET_PATHS: list[Path] = sorted(PRESET_DIRECTORY.glob("*/*.yaml"))


def test_every_backend_has_presets() -> None:
    assert {path.parent.name for path in PRESET_PATHS} == {backend.value for backend in ClassifierBackend}
    assert sum(len(PresetCatalog.names(backend)) for backend in ClassifierBackend) == len(PRESET_PATHS)


@pytest.mark.parametrize("path", PRESET_PATHS, ids=lambda path: path.stem)
def test_preset_loads_every_field(path: Path) -> None:
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    section: object = document["classifier"]
    assert isinstance(section, dict)
    assert set(section) == {field.name for field in fields(ClassifierSettings)}
    settings: ClassifierSettings = PresetCatalog.load(ClassifierBackend(path.parent.name), path.stem)
    assert settings.backend.value == path.parent.name
    assert settings.image_pooling is settings.backend.trained_image_pooling
    assert settings == PresetCatalog.load_file(path)


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError, match="has no preset"):
        PresetCatalog.load(ClassifierBackend.CLIP, "v2_base_patch16_224")
    with pytest.raises(FileNotFoundError):
        PresetCatalog.load_file(Path("missing.yaml"))


@pytest.mark.parametrize(
    ("document", "error", "message"),
    [
        ("- a\n- b\n", TypeError, "must hold a mapping"),
        ("model: {}\n", KeyError, "'classifier' mapping"),
    ],
)
def test_malformed_settings_files_raise(tmp_path: Path, document: str, error: type[Exception], message: str) -> None:
    path: Path = tmp_path / "settings.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(error, match=message):
        PresetCatalog.load_file(path)


def test_from_mapping_fills_defaults() -> None:
    settings: ClassifierSettings = ClassifierSettings.from_mapping(
        {"backend": "siglip", "weights_path": "siglip", "image_pooling": "patch_mean", "precision": "bfloat16"}
    )
    assert settings == ClassifierSettings(
        backend=ClassifierBackend.SIGLIP,
        weights_path="siglip",
        image_pooling=ImagePooling.PATCH_MEAN,
        precision=Precision.BFLOAT16,
    )
    assert settings.device is Device.AUTO


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"backend": "clip", "image_pooling": "class_token"}, KeyError, "weights_path"),
        (
            {"backend": "clip", "weights_path": "clip", "image_pooling": "class_token", "pooling": "x"},
            KeyError,
            "pooling",
        ),
        ({"backend": "vit", "weights_path": "clip", "image_pooling": "class_token"}, ValueError, "backend must be one"),
        ({"backend": "clip", "weights_path": 3, "image_pooling": "class_token"}, TypeError, "weights_path"),
        (
            {"backend": "clip", "weights_path": "clip", "image_pooling": "class_token", "batch_size": True},
            TypeError,
            "batch_size",
        ),
        (
            {"backend": "clip", "weights_path": "clip", "image_pooling": "class_token", "text_templates": "{}"},
            TypeError,
            "text_templates",
        ),
        (
            {"backend": "clip", "weights_path": "clip", "image_pooling": "class_token", "text_templates": ["{}", 1]},
            TypeError,
            "text_templates",
        ),
        ({"backend": "clip", "weights_path": "clip", "image_pooling": "attention"}, ValueError, "does not offer"),
    ],
)
def test_from_mapping_rejects_invalid_values(values: dict[str, object], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        ClassifierSettings.from_mapping(values)


def test_backend_capabilities() -> None:
    assert ClassifierBackend.CLIP.supported_image_poolings == frozenset(
        {ImagePooling.CLASS_TOKEN, ImagePooling.PATCH_MEAN}
    )
    assert ClassifierBackend.SIGLIP.supported_image_poolings == frozenset(
        {ImagePooling.ATTENTION, ImagePooling.PATCH_MEAN}
    )
    for backend in ClassifierBackend:
        assert backend.trained_image_pooling in backend.supported_image_poolings
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
