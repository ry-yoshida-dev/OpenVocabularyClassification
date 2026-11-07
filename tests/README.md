# tests

## Overview

Unit tests that run without downloading model weights. Tests marked `pretrained` download real weights and are skipped unless `--run-pretrained` is given.

## Components

| Component | Description |
| --------- | ----------- |
| [conftest.py](./conftest.py) | Tiny randomly initialized checkpoints and the `--run-pretrained` option. |
| [test_backends.py](./test_backends.py) | Backends compared with the `transformers` models on tiny checkpoints. |
| [test_pretrained.py](./test_pretrained.py) | Backends with real weights. |
| [test_cli.py](./test_cli.py) | Command-line prompt parsing, image discovery, text and JSON lines output, aggregation steps, preset listing and argument errors. |
| [test_image_tiling.py](./test_image_tiling.py) | Tile sizes, overlap, order and validation. |
| [test_letterbox.py](./test_letterbox.py) | Centered square padding, square images unchanged and fill color validation. |
| [test_uniform_attention_pooling.py](./test_uniform_attention_pooling.py) | Patch mean through the MAP head equal to the head with uniform attention, padding excluded. |
| [test_classifier.py](./test_classifier.py) | Shared classifier behavior: batching, streaming, tiling, image conversion and scoring. |
| [test_classification_result.py](./test_classification_result.py) | `ClassificationResult`, aggregation steps and scores. |
| [test_query_embedding_store.py](./test_query_embedding_store.py) | `QueryEmbeddingStore` caching and assembly. |
| [test_least_recently_used.py](./test_least_recently_used.py) | Bounded mapping eviction order, refresh on access and capacity validation. |
| [test_prompt.py](./test_prompt.py) | Prompts with text, visual and mixed queries: validation, kinds and query-to-class mapping. |
| [test_text_templates.py](./test_text_templates.py) | Template filling and placeholder validation. |
| [test_configuration.py](./test_configuration.py) | Presets loaded by `PresetCatalog`, `from_mapping` defaults and validation, backend capabilities and settings validation. |
| [test_torch_runtime.py](./test_torch_runtime.py) | `TorchRuntime` device resolution, precision, dtype and input transfer. |
| [test_package_import.py](./test_package_import.py) | Importing the package loads no backend library. |
