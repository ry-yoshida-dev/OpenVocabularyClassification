# options

## Overview

Options composing `ClassifierSettings`, shared by every backend.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `ClassifierBackend`, the classifier family and its capabilities. |
| [image_pooling.py](./image_pooling.py) | `ImagePooling`: class token, attention pooling or global average pooling of patch tokens. |
| [image_fit.py](./image_fit.py) | `ImageFit`: `letterbox` or `native`. |
| [prompt_aggregation.py](./prompt_aggregation.py) | `PromptAggregation`, the steps merging the embeddings of every class into candidates. |
| [aggregation_step.py](./aggregation_step.py) | `AggregationStep`, the unit, timing and method of one step. |
| [aggregation_unit.py](./aggregation_unit.py) | `AggregationUnit`: `template`, `prompt` or `template_and_prompt`. |
| [aggregation_timing.py](./aggregation_timing.py) | `AggregationTiming`: `embedding` or `similarity`. |
| [aggregation_method.py](./aggregation_method.py) | `AggregationMethod`: `mean`, `max`, `median`, `min` or `logsumexp`. |
| [score_activation.py](./score_activation.py) | `ScoreActivation`: softmax over classes (CLIP) or independent sigmoids (SigLIP). |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`. |
| [precision.py](./precision.py) | `Precision`: `float32`, `float16` or `bfloat16`. |

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
