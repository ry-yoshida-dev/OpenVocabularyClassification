# siglip

## Overview

SigLIP and SigLIP 2 through `transformers` (`ClassifierBackend.SIGLIP`). `SiglipArchitecture` picks the classifier
from the checkpoint's `model_type`: `siglip` covers SigLIP and the fixed-resolution SigLIP 2 weights (square input),
`siglip2` covers the SigLIP 2 NaFlex weights, which keep the aspect ratio of every image and pad shorter patch
sequences within a mini-batch.

The vision tower has no class token; a multihead attention pooling head (MAP) turns the normalized patch tokens into
the image embedding:

| `ImagePooling` | Image embedding |
| -------------- | --------------- |
| `ATTENTION` | The MAP head: a learned probe attends to every patch, then a residual MLP; as SigLIP was trained. |
| `PATCH_MEAN` | The mean of the patch tokens sent through the MAP head with uniform attention weights (`UniformAttentionPooling`). |

Uniform weights make the attention equal to the value and output projections of the mean token, since both are affine
and the weights of each head sum to one, so the mean reaches the joint space through the same layers as the learned
pooling. NaFlex padding is excluded from both poolings, so an embedding does not depend on the other images of its
mini-batch.

Text is lowercased (the SigLIP 2 Gemma tokenizer does not lowercase on its own), padded to the full text length
(64 tokens) and encoded without an attention mask, as the text towers were trained. Logits include the learned bias
and are scored with an independent sigmoid per class. SigLIP was trained on captions, so short phrases often get low
absolute probabilities even when they rank first; compare logits or ranks, or choose thresholds on real data.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `SiglipFamilyClassifier`: image poolings, lowercased unmasked text encoding, logit scale and bias. |
| [classifier/](./classifier/README.md) | `SiglipClassifier` (`siglip`) and `Siglip2NaFlexClassifier` (`siglip2`). |
| [architecture.py](./architecture.py) | `SiglipArchitecture`: architecture of a checkpoint and the classifier that loads it. |
| [uniform_attention_pooling.py](./uniform_attention_pooling.py) | `UniformAttentionPooling`: patch mean (padding excluded) carried through the MAP head. |
| [vision_tokens.py](./vision_tokens.py) | `SiglipVisionTokens`: normalized patch tokens, attention-pooled embeddings and patch mask of a vision tower run. |
