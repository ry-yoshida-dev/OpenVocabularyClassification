# siglip

## Overview

SigLIP and SigLIP 2 through `transformers` (`ClassifierBackend.SIGLIP`): `siglip` covers SigLIP and fixed-resolution SigLIP 2, and `siglip2` SigLIP 2 NaFlex, which keeps the aspect ratio of every image. Images are pooled by the attention pooling head or the patch mean, and scored with an independent sigmoid per class.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `SiglipFamilyClassifier`, the shared base of SigLIP-family models. |
| [classifier/](./classifier/README.md) | `SiglipClassifier` (`siglip`) and `Siglip2NaFlexClassifier` (`siglip2`). |
| [architecture.py](./architecture.py) | `SiglipArchitecture`, the architecture of a checkpoint. |
| [uniform_attention_pooling.py](./uniform_attention_pooling.py) | `UniformAttentionPooling`, the patch mean carried into the joint space. |
| [vision_tokens.py](./vision_tokens.py) | `SiglipVisionTokens`, the outputs of a vision tower run. |
