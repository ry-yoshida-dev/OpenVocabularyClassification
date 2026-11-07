# open_vocabulary_classification

## Overview

Model-independent interface, prompt, settings and result types for open-vocabulary image classification, the YAML
presets of every supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [classifier.py](./classifier.py) | `OpenVocabularyClassifier`, the runtime-independent base classifying images and embeddings and embedding images, texts and prompts. |
| [prepared_prompt.py](./prepared_prompt.py) | `PreparedPrompt`, the embeddings and logit aggregation of a prompt. |
| [letterbox.py](./letterbox.py) | `Letterbox`, which pads images to a centered square. |
| [image_tiling.py](./image_tiling.py) | `ImageTiling`, an overlapping tile grid for finding small objects. |
| [settings/](./settings/README.md) | `ClassifierSettings` and `PresetCatalog`. |
| [options/](./options/README.md) | Options composing `ClassifierSettings`. |
| [prompt/](./prompt/README.md) | `Prompt`, its query types, and text templates. |
| [result/](./result/README.md) | `ClassificationResult` and the scoring of logits. |
| [cache/](./cache/README.md) | Prompt embedding caches. |
| [runtime/](./runtime/README.md) | Execution of PyTorch-backed models. |
| [backends/](./backends/README.md) | Concrete classifiers, one sub-package per backend. |
| [config/](./config/README.md) | YAML presets of every backend, loaded by `PresetCatalog`. |
| [cli/](./cli/README.md) | The `open-vocabulary-classify` command. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |

## Examples

```python
classifier = PresetCatalog.load(ClassifierBackend.CLIP, "openai_vit_b16").build()

result = classifier.classify(image, Prompt.from_class_names(("cat", "dog")))
tiled_results = classifier.classify_images(images, prompt, ImageTiling(grid_size=3))
for result in classifier.iter_classify_images(lazily_loaded_images, prompt):
    ...
```
