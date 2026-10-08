# options

## Overview

Individual options composing `ClassifierSettings`, shared by every backend: which classifier family runs, how its
image embedding is pooled, where and in which floating-point precision it runs, and how its logits become scores.

## Components

| Component | Description |
| --------- | ----------- |
| [backend.py](./backend.py) | `ClassifierBackend`: classifier family, the image poolings it offers, the pooling it was trained with and its score activation. |
| [image_pooling.py](./image_pooling.py) | `ImagePooling`: class token, attention pooling or global average pooling of patch tokens. |
| [score_activation.py](./score_activation.py) | `ScoreActivation`: softmax over classes (CLIP) or independent sigmoids (SigLIP). |
| [device.py](./device.py) | `Device`: `auto`, `cpu`, `cuda` or `mps`; each runtime resolves it to its own device. |
| [precision.py](./precision.py) | `Precision`: `float32`, `float16` (GPU only) or `bfloat16`; each runtime resolves it to its own dtype. |

## Examples

```python
ClassifierBackend.SIGLIP.supported_image_poolings  # {ATTENTION, PATCH_MEAN}
ClassifierBackend.SIGLIP.trained_image_pooling  # ATTENTION
ClassifierBackend.CLIP.score_activation  # SOFTMAX
ImagePooling.PATCH_MEAN
Device.CUDA
Precision.BFLOAT16
```
