# options

## Overview

Individual options composing `ClassifierSettings`, shared by every backend: which classifier family runs, how its
image embedding is pooled, how images are fitted into the model input, where and in which floating-point precision it runs, how the sentences of a text query and the queries of a class are
combined, and how its logits become scores.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `ClassifierBackend`: classifier family, the image poolings it offers, the pooling it was trained with and its score activation. |
| [image_pooling.py](./image_pooling.py) | `ImagePooling`: class token, attention pooling or global average pooling of patch tokens. |
| [image_fit.py](./image_fit.py) | `ImageFit`: `letterbox` pads images to a square to keep them whole, `native` uses the checkpoint's own preprocessing. |
| [template_combination.py](./template_combination.py) | `TemplateCombination`: `ensemble` averages the sentence embeddings of a text query into one, `separate` keeps every sentence as a candidate, `max`, `mean`, `median` or `min` reduce the sentence logits of the query. |
| [logit_combination.py](./logit_combination.py) | `LogitCombination`: `max`, `mean`, `median` or `min` of several logits, e.g. the candidates of a class. |
| [score_activation.py](./score_activation.py) | `ScoreActivation`: softmax over classes (CLIP) or independent sigmoids (SigLIP). |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`; each runtime resolves it to its own device. |
| [precision.py](./precision.py) | `Precision`: `float32`, `float16` (GPU only) or `bfloat16`; each runtime resolves it to its own dtype. |

## Examples

```python
ClassifierBackend.SIGLIP.supported_image_poolings  # {ATTENTION, PATCH_MEAN}
ClassifierBackend.SIGLIP.trained_image_pooling  # ATTENTION
ClassifierBackend.CLIP.score_activation  # SOFTMAX
ImagePooling.PATCH_MEAN
ImageFit.LETTERBOX
TemplateCombination.MEAN.logit_combination  # LogitCombination.MEAN
LogitCombination.MEDIAN
Device.CUDA
Precision.BFLOAT16
```
