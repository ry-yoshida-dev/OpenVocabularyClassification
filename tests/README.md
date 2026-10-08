# tests

## Overview

Unit tests that run without downloading model weights. Backend tests build tiny randomly initialized CLIP, MetaCLIP 2,
SigLIP and SigLIP 2 NaFlex checkpoints (`conftest.py`) and compare the classifiers with the `transformers` models.

Tests marked `pretrained` download real weights and are skipped unless `--run-pretrained` is given; CI runs them
weekly and on manual dispatch.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Session fixture saving tiny checkpoints with their processors and tokenizers; the `--run-pretrained` option. |
| [test_backends.py](./test_backends.py) | Architecture selection, native pooling, logits and text embeddings equal to the `transformers` models, CLIP patch mean, NaFlex padding invariance, visual queries, classification from embeddings, tiling, bfloat16 and text length limits. |
| [test_pretrained.py](./test_pretrained.py) | Real CLIP, SigLIP, SigLIP 2 and SigLIP 2 NaFlex weights: embeddings and logits equal to `transformers`, solid colors named and found by text search. |
| [test_cli.py](./test_cli.py) | Command-line prompt parsing, image discovery, text and JSON lines output, preset listing and argument errors. |
| [test_image_tiling.py](./test_image_tiling.py) | Tile sizes, overlap, order and validation. |
| [test_uniform_attention_pooling.py](./test_uniform_attention_pooling.py) | Patch mean through the MAP head equal to the head with uniform attention, padding excluded. |
| [test_classifier.py](./test_classifier.py) | Shared mini-batching, streaming, tiling, classification from embeddings, text and prompt embeddings, RGB conversion with transparency over white, scoring and output shape checks. |
| [test_classification_result.py](./test_classification_result.py) | Best query per class, softmax and sigmoid scores, ranking, score and logit filtering, lookup, records and read-only arrays. |
| [test_query_embedding_store.py](./test_query_embedding_store.py) | Template ensembling, caching per text query and reference, least-recently-used eviction, reference averaging, release, query order and concurrent use. |
| [test_least_recently_used.py](./test_least_recently_used.py) | Bounded mapping eviction order, refresh on access and capacity validation. |
| [test_prompt.py](./test_prompt.py) | Prompts with text, visual and mixed queries: validation, kinds and query-to-class mapping. |
| [test_text_templates.py](./test_text_templates.py) | Template filling and placeholder validation. |
| [test_configuration.py](./test_configuration.py) | Presets loaded by `PresetCatalog`, `from_mapping` defaults and validation, backend capabilities and settings validation. |
| [test_torch_runtime.py](./test_torch_runtime.py) | `TorchRuntime` device resolution, precision, dtype and input transfer. |
| [test_package_import.py](./test_package_import.py) | Importing the package loads no backend library. |
