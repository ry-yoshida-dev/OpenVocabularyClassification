# config

## Overview

Ready-made YAML presets of every supported model, one folder per `ClassifierBackend` value, named by source, model
size and input resolution (e.g. `openai_vit_l14_336`, `v2_so400m_patch16_naflex`).

Each preset's `classifier` section maps one-to-one onto the fields of
[`ClassifierSettings`](../settings/core.py) (enums written by value, `text_templates` as a list).
[`PresetCatalog`](../settings/presets.py) loads a preset by backend and name, or any YAML file in this format;
`ClassifierSettings.from_mapping` or any dataclass builder reads the section from other configuration systems.
Switching models means switching preset files. Every preset uses the pooling its model was trained with (`class_token` for CLIP, `attention` for SigLIP);
set `image_pooling: patch_mean` to use global average pooling of the patch tokens instead.

| Folder | Presets |
| ------ | ------- |
| [clip/](./clip) | OpenAI CLIP `openai_vit_b32`, `openai_vit_b16`, `openai_vit_l14`, `openai_vit_l14_336`; OpenCLIP LAION-2B `laion_vit_b32`, `laion_vit_h14`, `laion_vit_bigg14`; MetaCLIP `metaclip_vit_b16`, `metaclip_vit_l14`, `metaclip_vit_h14`; MetaCLIP 2 `metaclip2_vit_h14_quickgelu`, `metaclip2_vit_h14_378` |
| [siglip/](./siglip) | SigLIP `v1_base_patch16_224`, `v1_large_patch16_384`, `v1_so400m_patch14_384`; SigLIP 2 `v2_base_patch16_224`, `v2_large_patch16_384`, `v2_so400m_patch14_384`, `v2_giant_opt_patch16_384`; SigLIP 2 NaFlex `v2_base_patch16_naflex`, `v2_so400m_patch16_naflex` |

## Examples

```yaml
classifier:
  backend: clip
  weights_path: openai/clip-vit-base-patch16
  image_pooling: class_token
  text_templates:
    - "a photo of a {}."
    - "a blurry photo of a {}."
    - "a close-up photo of a {}."
  batch_size: 32
  device: auto
  precision: float32
```

```python
settings = PresetCatalog.load(ClassifierBackend.CLIP, "openai_vit_b16")
```

A preset is a starting point: copy it into an application config to change the pooling, templates, device or
precision (`bfloat16` keeps large models such as `v2_giant_opt_patch16_384` stable at half the memory), or point
`weights_path` at fine-tuned or local weights of the same backend.
