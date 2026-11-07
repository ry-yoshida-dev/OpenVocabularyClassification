# clip

## Overview

CLIP-architecture models through `transformers` (`ClassifierBackend.CLIP`): OpenAI CLIP, OpenCLIP weights converted to `transformers` (LAION, DataComp, Apple DFN) and MetaCLIP (`model_type` `clip`), and MetaCLIP 2 (`metaclip_2`). Images are pooled by the class token or the patch mean, and scored with a softmax over the classes.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `ClipFamilyClassifier`, the shared base of CLIP-architecture models. |
| [classifier/](./classifier/README.md) | `ClipClassifier` (`clip`) and `MetaClip2Classifier` (`metaclip_2`). |
| [architecture.py](./architecture.py) | `ClipArchitecture`, the architecture of a checkpoint. |
