# backends

## Overview

Concrete `OpenVocabularyClassifier` implementations. Every classifier takes the common `ClassifierSettings`, declares
the `ClassifierBackend` it serves, and is a `DualEncoderClassifier`: an image tower and a text tower embed into one
space, and the logit of an image and a query is their cosine similarity times the learned scale, plus a learned bias
for SigLIP. Each family implements its image poolings and text encoding; templates, caching, mini-batching, tiling and
scoring are shared.

A backend groups the models sharing one implementation, and the `model_type` of the checkpoint (`CheckpointConfig`)
chooses the classifier, so `weights_path` alone decides the architecture.

Nothing is imported eagerly: `ClassifierSettings.build()` imports only the sub-package of the requested backend, so
importing `open_vocabulary_classification` does not load `transformers`. Import a classifier class directly from its
sub-package (e.g. `backends.siglip`) when needed.

## Components

| Component | Description |
| --------- | ----------- |
| [dual_encoder.py](./dual_encoder.py) | `DualEncoderClassifier`: model loading, logit scale and bias read once in `float32` before the precision cast, cached text, prompt and image embeddings, padded tokenization with a length check. |
| [checkpoint_config.py](./checkpoint_config.py) | `CheckpointConfig`: reads the `model_type` of a `transformers` checkpoint without loading its weights. |
| [clip/](./clip/README.md) | CLIP, OpenCLIP, MetaCLIP and MetaCLIP 2; class token or patch mean, softmax scores. |
| [siglip/](./siglip/README.md) | SigLIP, SigLIP 2 and SigLIP 2 NaFlex; attention pooling or patch mean, sigmoid scores. |
