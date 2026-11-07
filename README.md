# OpenVocabularyClassification

## Overview

`open_vocabulary_classification` puts CLIP-family and SigLIP-family image classifiers behind one interface, so that models can be swapped and compared by changing configuration only. Application code builds a `Prompt` that maps each class name to text phrases (`TextQuery`), reference images (`VisualQuery`) or both, calls one `OpenVocabularyClassifier`, and reads one `ClassificationResult` type. Which model runs is decided by a `ClassifierSettings`, normally loaded from a YAML [preset](src/open_vocabulary_classification/config/README.md) by `PresetCatalog`.

The classifier also returns the embeddings it scores, for text-to-image search or for classifying stored embeddings without running the image tower again. The `open-vocabulary-classify` [command](src/open_vocabulary_classification/cli/README.md) classifies files and directories.

| Backend | Models | Image poolings | Scores | Presets |
| ------- | ------ | -------------- | ------ | ------- |
| `clip` | OpenAI CLIP, OpenCLIP (LAION), MetaCLIP, MetaCLIP 2, Apple DFN CLIP | `CLASS_TOKEN` (trained), `PATCH_MEAN` | Softmax over classes | `openai_vit_b32` ... `openai_vit_l14_336`, `laion_vit_b32` ... `laion_vit_bigg14`, `metaclip_vit_b16` ... `metaclip_vit_h14`, `metaclip2_vit_h14_*`, `dfn2b_vit_l14*`, `dfn5b_vit_h14*` |
| `siglip` | SigLIP, SigLIP 2, SigLIP 2 NaFlex | `ATTENTION` (trained), `PATCH_MEAN` | Independent sigmoid per class | `v1_*`, `v2_*_224` ... `v2_giant_opt_patch16_384`, `v2_*_naflex` |

## Installation

```bash
pip install .
```

## Usage

Load a packaged preset (or any YAML file in the same format with `PresetCatalog.load_file`), then run any model with the same code:

```python
from open_vocabulary_classification import ClassifierBackend, PresetCatalog, Prompt

classifier = PresetCatalog.load(ClassifierBackend.SIGLIP, "v2_base_patch16_naflex").build()

results = classifier.classify_images(images, Prompt.from_class_names(("cat", "dog", "bus")))
for classification in results[0].top_k(3):
    print(classification.class_name, classification.score, classification.logit)
```

Embed once, then search by text and classify with several prompts without running the image tower again:

```python
embeddings = classifier.embed_images(images)
similarities = embeddings @ classifier.embed_texts(["a dog on a beach"]).T
scenes = classifier.classify_embeddings(embeddings, Prompt.from_class_names(("indoor", "outdoor")))
```

Several phrases and reference images for one class; `matched_query` tells which query matched:

```python
from open_vocabulary_classification import Prompt, TextQuery, VisualQuery, VisualReference

prompt = Prompt(
    {
        "dog": (TextQuery("dog"), TextQuery("puppy")),
        "my dog": (VisualQuery((VisualReference(my_dog_photo_1), VisualReference(my_dog_photo_2))),),
    }
)
result = classifier.classify(image, prompt)
print(result.top.class_name, result.top.matched_query)
```

From the command line:

```bash
open-vocabulary-classify images/ --preset clip/openai_vit_b16 --classes cat "dog:dog,puppy" bus
```

## Development

Tests use tiny random checkpoints by default; `--run-pretrained` also compares real CLIP and SigLIP weights with `transformers`:

```bash
pytest --run-pretrained
```
