# backends

## Overview

Concrete `OpenVocabularyClassifier` implementations, one sub-package per `ClassifierBackend`; the checkpoint's `model_type` selects the classifier within a backend. `ClassifierSettings.build()` imports only the sub-package of the requested backend, so importing `open_vocabulary_classification` does not load `transformers`.

## Components

| Component | Description |
| --------- | ----------- |
| [dual_encoder.py](./dual_encoder.py) | `DualEncoderClassifier`, the shared base of image-text dual encoders. |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`, which reads a checkpoint's configuration without loading its weights. |
| [clip/](./clip/README.md) | CLIP, OpenCLIP, MetaCLIP, MetaCLIP 2 and DFN CLIP. |
| [siglip/](./siglip/README.md) | SigLIP, SigLIP 2 and SigLIP 2 NaFlex. |
