# examples

## Overview

Runnable scripts using the package. To classify a directory from the command line, use the installed
`open-vocabulary-classify` command (see [cli/](../src/open_vocabulary_classification/cli/README.md)).

## Components

| Component | Description |
| --------- | ----------- |
| [search_images.py](./search_images.py) | Embeds a directory once with `iter_embed_images`, searches it by text with `embed_texts`, and classifies the same embeddings with two prompts through `classify_embeddings`. |

## Examples

```bash
python examples/search_images.py images/ --preset siglip/v2_base_patch16_224 --query "a dog on a beach" --top-k 5
```
