# examples

## Overview

Runnable scripts using the package.

## Components

| Component | Description |
| --------- | ----------- |
| [classify_directory.py](./classify_directory.py) | Classifies every image of a directory with any backend, weights, pooling and templates, reading one mini-batch of EXIF-oriented images at a time; `name:query,query` gives a class several text queries, and a repeated class collects the queries of every argument. |

## Examples

```bash
python examples/classify_directory.py images/ --backend clip --weights openai/clip-vit-base-patch16 --classes cat dog bus
python examples/classify_directory.py images/ --backend siglip --weights google/siglip2-base-patch16-naflex \
    --pooling patch_mean --templates "a photo of a {}." "a close-up photo of a {}." --classes "dog:dog,puppy" cat
```
