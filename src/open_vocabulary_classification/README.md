# open_vocabulary_classification

## Overview

Model-independent interface, prompt, settings and result types for open-vocabulary image classification, the YAML
presets of every supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [classifier.py](./classifier.py) | `OpenVocabularyClassifier`: runtime-independent base classifying images (eagerly, streamed or tiled) and precomputed embeddings, embedding images, texts and prompts, with mini-batching, RGB conversion over a white background (stretching high-bit-depth images), tiled views batched across images, and shared scoring. |
| [image_tiling.py](./image_tiling.py) | `ImageTiling`: overlapping tile grid, plus the whole image, classified view by view so small objects are found. |
| [settings/](./settings/README.md) | `ClassifierSettings` with `from_mapping` and `build()`, and `PresetCatalog` loading packaged presets and YAML files. |
| [options/](./options/README.md) | `ClassifierBackend`, `ImagePooling`, `ScoreActivation`, `Device` and `Precision`: options composing `ClassifierSettings`. |
| [prompt/](./prompt/README.md) | `Prompt` mapping each class name to its text and visual queries, the query types, `TextTemplates` and `VisualReference`. |
| [result/](./result/README.md) | `ClassificationResult`, `Classification` and its JSON-serializable `ClassificationRecord`. |
| [cache/](./cache/README.md) | Thread-safe query embeddings cached per text query and per visual reference. |
| [runtime/](./runtime/README.md) | `TorchRuntime`: device, precision, model preparation and input transfer for PyTorch-backed classifiers. |
| [backends/](./backends/README.md) | CLIP-family (CLIP, OpenCLIP, MetaCLIP, MetaCLIP 2) and SigLIP-family (SigLIP, SigLIP 2, SigLIP 2 NaFlex) classifiers. |
| [config/](./config/README.md) | YAML presets of every backend, loaded by `PresetCatalog`. |
| [cli/](./cli/README.md) | `open-vocabulary-classify` command classifying image files and directories by text and reference images. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |

## Examples

```python
classifier = PresetCatalog.load(ClassifierBackend.CLIP, "openai_vit_b16").build()

result = classifier.classify(image, Prompt.from_class_names(("cat", "dog")))
results = classifier.classify_images(images, Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)}))
tiled_results = classifier.classify_images(images, prompt, ImageTiling(grid_size=3))
for result in classifier.iter_classify_images(lazily_loaded_images, prompt):
    ...

embeddings = classifier.embed_images(images)
results = classifier.classify_embeddings(embeddings, prompt)
tiled_results = classifier.classify_embeddings(classifier.embed_images(images, ImageTiling()), prompt)
similarities = embeddings @ classifier.embed_texts(["a dog on a beach"]).T
```
