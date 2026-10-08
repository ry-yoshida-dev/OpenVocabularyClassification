# settings

## Overview

Everything needed to load and run a classifier, identical for every backend, and the readers turning configuration
files into it. `ClassifierSettings` validates its fields on construction and `build()` loads the classifier of its
backend. `ClassifierSettings.from_mapping` reads plain values (enums by value), so any parsed YAML or JSON section
works; `PresetCatalog` reads the presets shipped under [config/](../config/README.md) and YAML files in the same
format. Malformed configuration (unknown or missing keys, wrong types, unknown enum values) fails before any model is
loaded.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `ClassifierSettings`: backend, weights, image pooling, text templates, batch size, device and precision; `from_mapping` and `build()`. |
| [presets.py](./presets.py) | `PresetCatalog`: lists and loads packaged presets (`clip/openai_vit_b16`) and preset-format YAML files. |
| [config_values.py](./config_values.py) | `ConfigValues`: typed, validating access to strings, string lists, integers and enum values of a configuration section. |

## Examples

```python
settings = PresetCatalog.load(ClassifierBackend.SIGLIP, "v2_base_patch16_naflex")
settings = PresetCatalog.load_file(Path("my_classifier.yaml"))
PresetCatalog.names(ClassifierBackend.CLIP)  # ("laion_vit_b32", ...)

settings = ClassifierSettings.from_mapping(
    {"backend": "clip", "weights_path": "openai/clip-vit-base-patch16", "image_pooling": "class_token"}
)
classifier = dataclasses.replace(settings, precision=Precision.BFLOAT16).build()
```
