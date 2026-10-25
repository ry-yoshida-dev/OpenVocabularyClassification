# options

## Overview

Individual options composing `ClassifierSettings`, shared by every backend: which classifier family runs, how its
image embedding is pooled, how images are fitted into the model input, where and in which floating-point precision it runs, how the embeddings of a class are
aggregated, and how its logits become scores.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `ClassifierBackend`: classifier family, the image poolings it offers, the pooling it was trained with and its score activation. |
| [image_pooling.py](./image_pooling.py) | `ImagePooling`: class token, attention pooling or global average pooling of patch tokens. |
| [image_fit.py](./image_fit.py) | `ImageFit`: `letterbox` pads images to a square to keep them whole, `native` uses the checkpoint's own preprocessing. |
| [prompt_aggregation.py](./prompt_aggregation.py) | `PromptAggregation`: the steps turning the prompt × template table of every class into one logit; one `template_and_prompt` step or a `template` and a `prompt` step, at non-decreasing timings, at most one at `embedding`. |
| [aggregation_step.py](./aggregation_step.py) | `AggregationStep`: unit, timing and method of one step; embeddings by `mean` only. |
| [aggregation_unit.py](./aggregation_unit.py) | `AggregationUnit`: `template`, `prompt` or `template_and_prompt`. |
| [aggregation_timing.py](./aggregation_timing.py) | `AggregationTiming`: `embedding` (before the image is compared) or `similarity` (before the score activation). |
| [aggregation_method.py](./aggregation_method.py) | `AggregationMethod`: `mean`, `max`, `median`, `min` or `logsumexp`. |
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
PromptAggregation()  # templates averaged at embedding, then the best prompt at similarity
AggregationStep(AggregationUnit.TEMPLATE_AND_PROMPT, AggregationTiming.SIMILARITY, AggregationMethod.LOGSUMEXP)
Device.CUDA
Precision.BFLOAT16
```
