# OpenVocabularyClassification

## Overview

`open_vocabulary_classification` puts CLIP, OpenCLIP, MetaCLIP, MetaCLIP 2, SigLIP and SigLIP 2 behind one interface
so that models can be swapped and compared by changing configuration only. Application code builds a prompt, calls one
`OpenVocabularyClassifier`, and reads one `ClassificationResult` type; which model runs is decided by a
`ClassifierSettings`, normally built from a YAML preset.

- **Configuration is data**: every model-specific value (weights, image pooling, text templates, batch size) lives in
  a YAML preset under [config/](src/open_vocabulary_classification/config/README.md), one folder per backend. The same
  `ClassifierSettings` fields apply to every model.
- **The image pooling is selectable**: besides the pooling each model was trained with (the class token of CLIP, the
  attention pooling head of SigLIP), every model offers `PATCH_MEAN`, the global average of its patch tokens, carried
  into the joint image-text space so that text queries still apply. `embed_images` returns the same embeddings for
  retrieval or clustering.
- **Classes are separate from queries**: a `Prompt` maps each class name (class id = key order) to its queries, text
  phrases (`TextQuery`), reference images (`VisualQuery`) or both, e.g. `"dog"` by `"dog"`, `"puppy"` and photos of
  the user's own dog. Each class takes the logit of its best query and reports it as matched.
- **Prompt ensembling is built in**: every text query is filled into the `text_templates` of the settings (e.g.
  `"a photo of a {}."`) and the normalized sentence embeddings are averaged.
- **Scores follow the training loss**: CLIP scores are a softmax over the prompt classes (exactly one class is
  assumed), SigLIP scores are independent sigmoids (any number of classes may apply). Raw logits are kept as well.

| Pooling | `clip` | `siglip` |
| ------- | :----: | :------: |
| `CLASS_TOKEN` | yes (trained) | - |
| `ATTENTION` | - | yes (trained) |
| `PATCH_MEAN` | yes | yes |

| Backend | Models | Architectures (`model_type`) | Presets |
| ------- | ------ | ---------------------------- | ------- |
| `clip` | OpenAI CLIP, OpenCLIP (LAION), MetaCLIP, MetaCLIP 2 | `clip`, `metaclip_2` | `openai_vit_b32` ... `openai_vit_l14_336`, `laion_vit_b32` ... `laion_vit_bigg14`, `metaclip_vit_b16` ... `metaclip_vit_h14`, `metaclip2_vit_h14_*` |
| `siglip` | SigLIP, SigLIP 2, SigLIP 2 NaFlex | `siglip`, `siglip2` | `v1_*`, `v2_*_224` ... `v2_giant_opt_patch16_384`, `v2_*_naflex` |

A backend groups the models that share one implementation; the `model_type` in the checkpoint's `config.json` selects
the architecture, so `weights_path` alone decides between them.

Images are processed in mini-batches; images with transparency are composited over white. Query embeddings are cached
per text query (the most recently used 4096) and per visual reference, so prompts that share queries do not recompute
them. Weights are downloaded on first use.

For module details, see [src/open_vocabulary_classification/README.md](src/open_vocabulary_classification/README.md).

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Examples

Build settings from a preset with [DictConfigHandler](https://github.com/ry-yoshida-dev/DictConfigHandler), then run
any model with the same code. DictConfigHandler (which brings in OmegaConf) is not a dependency of this package;
install it separately:

```bash
pip install "dictconfig-handler @ git+https://github.com/ry-yoshida-dev/DictConfigHandler.git"
```

```python
from dictconfig_handler import DictConfigHandler
from omegaconf import OmegaConf

from open_vocabulary_classification import ClassifierSettings, Prompt

preset_path = "src/open_vocabulary_classification/config/siglip/v2_base_patch16_naflex.yaml"
settings = DictConfigHandler(cfg=OmegaConf.load(preset_path)).build_dataclass(ClassifierSettings, key="classifier")
classifier = settings.build()

results = classifier.classify_images(images, Prompt.from_class_names(("cat", "dog", "bus")))
for classification in results[0].top_k(3):
    print(classification.class_name, classification.score, classification.logit)
```

Global average pooling instead of the trained pooling, with several templates:

```python
from open_vocabulary_classification import ClassifierBackend, ClassifierSettings, ImagePooling

classifier = ClassifierSettings(
    backend=ClassifierBackend.CLIP,
    weights_path="openai/clip-vit-base-patch16",
    image_pooling=ImagePooling.PATCH_MEAN,
    text_templates=("a photo of a {}.", "a close-up photo of a {}.", "a cropped photo of a {}."),
).build()

embeddings = classifier.embed_images(images)
similarities = embeddings @ embeddings.T
```

Several phrases and reference images for one class: the class is reported as `"dog"`, and
`classification.matched_query` tells which query matched. The reference images of one `VisualQuery` are averaged into
one query, so group photos of the same appearance and give different appearances separate queries.

```python
from open_vocabulary_classification import Prompt, TextQuery, VisualQuery, VisualReference

prompt = Prompt(
    {
        "dog": (TextQuery("dog"), TextQuery("puppy")),
        "my dog": (VisualQuery((VisualReference(my_dog_photo_1), VisualReference(my_dog_photo_2))),),
        "cat": (TextQuery("cat"),),
    }
)
result = classifier.classify(image, prompt)
print(result.top.class_name, result.top.matched_query)
```

Text and visual logits come from different distributions (image-text versus image-image similarity), so check scores
on real data when mixing them. SigLIP was trained on captions, so short phrases often get low absolute probabilities
even when they rank first; use `filter_by_score` thresholds chosen on real data, or compare logits.

Command-line example over a directory:

```bash
python examples/classify_directory.py images/ --backend clip --weights openai/clip-vit-base-patch16 --classes cat "dog:dog,puppy" bus
```
