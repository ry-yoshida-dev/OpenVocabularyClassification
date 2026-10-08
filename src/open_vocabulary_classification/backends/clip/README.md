# clip

## Overview

CLIP-architecture models through `transformers` (`ClassifierBackend.CLIP`). `ClipArchitecture` picks the classifier
from the checkpoint's `model_type`: `clip` covers OpenAI CLIP, OpenCLIP weights converted to `transformers` (LAION,
DataComp) and MetaCLIP; `metaclip_2` covers MetaCLIP 2.

The vision tower prepends a class token to the patch tokens and ends in a layer norm and a linear projection:

| `ImagePooling` | Image embedding |
| -------------- | --------------- |
| `CLASS_TOKEN` | `visual_projection(post_layernorm(class_token))`, as CLIP was trained. |
| `PATCH_MEAN` | `visual_projection(mean(post_layernorm(patch_tokens)))`, global average pooling with the class token excluded. |

Text is padded to the full context of the text tower (77 tokens) and a text query filled into a template must fit in
it. Logits have no bias and are scored with a softmax over the prompt classes.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `ClipFamilyClassifier`: image poolings, text encoding and logit scale shared by every CLIP-architecture model. |
| [classifier/](./classifier/README.md) | `ClipClassifier` (`clip`) and `MetaClip2Classifier` (`metaclip_2`). |
| [architecture.py](./architecture.py) | `ClipArchitecture`: architecture of a checkpoint and the classifier that loads it. |
