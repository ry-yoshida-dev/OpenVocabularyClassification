# settings

## Overview

Everything needed to load and run a classifier, identical for every backend, and the readers turning configuration into it: `ClassifierSettings.from_mapping` for any parsed YAML or JSON section, and `PresetCatalog` for the presets under [config/](../config/README.md) and YAML files in the same format.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `ClassifierSettings`, everything needed to load and run a classifier. |
| [presets.py](./presets.py) | `PresetCatalog`, which lists and loads presets and preset-format YAML files. |
| [config_values.py](./config_values.py) | `ConfigValues`, typed access to the values of a configuration section. |

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
