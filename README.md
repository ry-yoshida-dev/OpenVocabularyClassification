# OpenVocabularyClassification

## Overview

`open_vocabulary_classification` puts CLIP, OpenCLIP, MetaCLIP, MetaCLIP 2, SigLIP and SigLIP 2 behind one interface
so that models can be swapped and compared by changing configuration only. Application code builds a prompt, calls one
`OpenVocabularyClassifier`, and reads one `ClassificationResult` type; which model runs is decided by a
`ClassifierSettings`, normally built from a YAML preset.

- **Configuration is data**: every model-specific value (weights, image pooling, text templates, batch size,
  precision) lives in a YAML preset under [config/](src/open_vocabulary_classification/config/README.md), one folder
  per backend, loaded by `PresetCatalog`. The same `ClassifierSettings` fields apply to every model.
- **The image pooling is selectable**: besides the pooling each model was trained with (the class token of CLIP, the
  attention pooling head of SigLIP), every model offers `PATCH_MEAN`, the global average of its patch tokens, carried
  into the joint image-text space so that text queries still apply.
- **Embeddings are first-class**: `embed_images` and `embed_texts` return the embeddings the classifier scores, for
  text-to-image search, retrieval or clustering, and `classify_embeddings` classifies stored image embeddings with any
  prompt without running the image tower again.
- **Classes are separate from queries**: a `Prompt` maps each class name (class id = key order) to its queries, text
  phrases (`TextQuery`), reference images (`VisualQuery`) or both, e.g. `"dog"` by `"dog"`, `"puppy"` and photos of
  the user's own dog. Each class takes the logit of its best query and reports it as matched.
- **Prompt ensembling is built in**: every text query is filled into the `text_templates` of the settings (e.g.
  `"a photo of a {}."`) and the normalized sentence embeddings are averaged.
- **Scores follow the training loss**: CLIP scores are a softmax over the prompt classes (exactly one class is
  assumed), SigLIP scores are independent sigmoids (any number of classes may apply). Raw logits are kept as well, and
  `filter_by_logit` thresholds them independently of the other classes. `to_records` serializes results as JSON.
- **Small objects can be found**: an `ImageTiling` classifies overlapping tiles at the full model resolution, plus the
  whole image, and keeps the best logit of each query. The views of several images share one mini-batch, and
  `embed_images(images, tiling)` keeps one embedding per view for `classify_embeddings`.

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

Images are processed in mini-batches, and `iter_classify_images` / `iter_embed_images` read a stream of images one
mini-batch at a time, so datasets of any size run in bounded memory; images with transparency are composited over
white, and 16-bit, 32-bit and float single-channel images are stretched from their lowest to their highest value.
Query embeddings are cached per text query (the most recently used 4096) and per visual reference, so prompts
that share queries do not recompute them; the caches are thread-safe. Models run in `float32`, `float16` (GPU) or
`bfloat16`; the learned logit scale and bias are always read in `float32`. Weights are downloaded on first use.

For module details, see [src/open_vocabulary_classification/README.md](src/open_vocabulary_classification/README.md).

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest                    # tiny random checkpoints, no download
pytest --run-pretrained   # also compares real CLIP and SigLIP weights with transformers
```

## Examples

Load a packaged preset (or any YAML file in the same format with `PresetCatalog.load_file`), then run any model with
the same code:

```python
from open_vocabulary_classification import ClassifierBackend, PresetCatalog, Prompt

classifier = PresetCatalog.load(ClassifierBackend.SIGLIP, "v2_base_patch16_naflex").build()

results = classifier.classify_images(images, Prompt.from_class_names(("cat", "dog", "bus")))
for classification in results[0].top_k(3):
    print(classification.class_name, classification.score, classification.logit)
```

`ClassifierSettings.from_mapping` builds settings from any parsed configuration section, e.g. one read by OmegaConf
or [DictConfigHandler](https://github.com/ry-yoshida-dev/DictConfigHandler).

Embed once, then search by text and classify with several prompts without running the image tower again:

```python
embeddings = classifier.embed_images(images)
similarities = embeddings @ classifier.embed_texts(["a dog on a beach"]).T

scenes = classifier.classify_embeddings(embeddings, Prompt.from_class_names(("indoor", "outdoor")))
times = classifier.classify_embeddings(embeddings, Prompt.from_class_names(("daytime", "night")))

view_embeddings = classifier.embed_images(images, ImageTiling(grid_size=2))
small_objects = classifier.classify_embeddings(view_embeddings, Prompt.from_class_names(("bird", "drone")))
```

Stream a large dataset, tile each image to find small objects, and write JSON lines:

```python
images = (load_image(path) for path in image_paths)
tiling = ImageTiling(grid_size=3, overlap_ratio=0.25)
for path, result in zip(image_paths, classifier.iter_classify_images(images, prompt, tiling)):
    print(json.dumps({"image": str(path), "classes": result.to_records()}))
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
even when they rank first; use `filter_by_score` thresholds chosen on real data, or compare logits. CLIP softmax
scores always favor some class, even for an image showing none of them; add a background class such as
`"something else"`, or reject images with `filter_by_logit`.

The `open-vocabulary-classify` command classifies files and directories from the command line:

```bash
open-vocabulary-classify --list-presets
open-vocabulary-classify images/ --preset clip/openai_vit_b16 --classes cat "dog:dog,puppy" bus
open-vocabulary-classify images/ --recursive --preset siglip/v2_so400m_patch14_384 --precision bfloat16 \
    --tile-grid 2 --classes cat dog --threshold 0.1 --format jsonl > results.jsonl
open-vocabulary-classify shelf/ --preset siglip/v2_base_patch16_224 --classes "my mug:mug,@refs/my_mug/" cup
```

A query `@path` uses the image file, or every image of the directory, at `path` as reference images of one visual
query. Files that cannot be decoded are reported on standard error and skipped; invalid arguments are reported before
the model is loaded.
