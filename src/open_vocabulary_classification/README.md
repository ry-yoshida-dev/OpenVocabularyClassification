# open_vocabulary_classification

## Overview

Model-independent interface, prompt, settings and result types for open-vocabulary image classification, the YAML
presets of every supported model, and the backends implementing the interface.

## Components

| Component | Description |
| --------- | ----------- |
| [classifier.py](./classifier.py) | `OpenVocabularyClassifier`: runtime-independent base with `classify`, `classify_images`, `embed_image`, `embed_images`, mini-batching, RGB conversion over a white background and shared scoring. |
| [settings.py](./settings.py) | `ClassifierSettings`: backend, weights, image pooling, text templates, batch size, device and precision; `build()` loads the classifier. |
| [options/](./options/README.md) | `ClassifierBackend`, `ImagePooling`, `ScoreActivation` and `Device`: options composing `ClassifierSettings`. |
| [prompt/](./prompt/README.md) | `Prompt` mapping each class name to its text and visual queries, the query types, `TextTemplates` and `VisualReference`. |
| [result/](./result/README.md) | `ClassificationResult` and `Classification`. |
| [cache/](./cache/README.md) | Query embeddings cached per text query and per visual reference. |
| [runtime/](./runtime/README.md) | `TorchRuntime`: device, precision, model preparation and input transfer for PyTorch-backed classifiers. |
| [backends/](./backends/README.md) | CLIP-family (CLIP, OpenCLIP, MetaCLIP, MetaCLIP 2) and SigLIP-family (SigLIP, SigLIP 2, SigLIP 2 NaFlex) classifiers. |
| [config/](./config/README.md) | YAML presets of every backend, buildable into `ClassifierSettings`. |
| [array_types.py](./array_types.py) | NumPy array aliases (`FloatArray`, `IntArray`, `BoolArray`). |

## Examples

```python
classifier = settings.build()

result = classifier.classify(image, Prompt.from_class_names(("cat", "dog")))
results = classifier.classify_images(images, Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)}))
embeddings = classifier.embed_images(images)
```
