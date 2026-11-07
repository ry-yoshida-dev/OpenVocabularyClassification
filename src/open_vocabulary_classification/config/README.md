# config

## Overview

Ready-made YAML presets of every supported model, one folder per `ClassifierBackend` value, named by source, model size and input resolution (e.g. `openai_vit_l14_336`, `v2_so400m_patch16_naflex`). Each preset's `classifier` section maps onto the fields of [`ClassifierSettings`](../settings/core.py) and uses the pooling its model was trained with and letterboxed images; copy a preset into an application config to change it.

Weights keep the license of their publisher and are downloaded from the Hugging Face Hub on first use. MetaCLIP and MetaCLIP 2 (CC BY-NC 4.0) and Apple DFN (Apple Machine Learning Research Model License) are for non-commercial research only; `dfn2b_vit_l14` was published with its untrained temperature, so its softmax scores are flat while its ranking is unaffected.

## Components

| Component | Presets |
| --------- | ------- |
| [clip/](./clip) | OpenAI CLIP `openai_vit_b32`, `openai_vit_b16`, `openai_vit_l14`, `openai_vit_l14_336`; OpenCLIP LAION-2B `laion_vit_b32`, `laion_vit_h14`, `laion_vit_bigg14`; MetaCLIP `metaclip_vit_b16`, `metaclip_vit_l14`, `metaclip_vit_h14`; MetaCLIP 2 `metaclip2_vit_h14_quickgelu`, `metaclip2_vit_h14_378`; Apple DFN `dfn2b_vit_l14`, `dfn2b_vit_l14_39b`, `dfn5b_vit_h14`, `dfn5b_vit_h14_378` |
| [siglip/](./siglip) | SigLIP `v1_base_patch16_224`, `v1_large_patch16_384`, `v1_so400m_patch14_384`; SigLIP 2 `v2_base_patch16_224`, `v2_large_patch16_384`, `v2_so400m_patch14_384`, `v2_giant_opt_patch16_384`; SigLIP 2 NaFlex `v2_base_patch16_naflex`, `v2_so400m_patch16_naflex` |

## Examples

```yaml
classifier:
  backend: clip
  weights_path: openai/clip-vit-base-patch16
  image_pooling: class_token
  image_fit: letterbox
  text_templates:
    - "a photo of a {}."
    - "a blurry photo of a {}."
    - "a close-up photo of a {}."
  aggregation:
    - {unit: template, timing: similarity, method: mean}
    - {unit: prompt, timing: similarity, method: max}
  batch_size: 32
  device: auto
  precision: float32
```

```python
settings = PresetCatalog.load(ClassifierBackend.CLIP, "openai_vit_b16")
```
