# cli

## Overview

The `open-vocabulary-classify` command, installed with the package. It classifies image files and directories into
classes given as arguments, queried by text phrases or by reference images (`@path`), with settings from a packaged
preset, a YAML file or a backend and weights, and prints the top classes of each image as text or JSON lines. Images
are read one mini-batch at a time, so directories of any size run in bounded memory; files that cannot be decoded are
reported on standard error and skipped. Invalid arguments are reported before the model is loaded.

## Components

| Component | Description |
| --------- | ----------- |
| [command.py](./command.py) | `ClassifyCommand`: argument parsing and validation, settings and prompt construction (text and `@path` visual queries), streaming classification and output. |
| [image_files.py](./image_files.py) | `ImageFiles`: expands files and directories into image files and reads them upright. |
| [image_file_stream.py](./image_file_stream.py) | `ImageFileStream`: reads image files lazily, skipping undecodable ones with a warning and tracking the path of each yielded image. |
| [output_format.py](./output_format.py) | `OutputFormat`: `text` for people or `jsonl` with one `ClassificationRecord` list per image. |

## Examples

```bash
open-vocabulary-classify --list-presets
open-vocabulary-classify images/ --preset clip/openai_vit_b16 --classes cat "dog:dog,puppy" bus
open-vocabulary-classify images/ --recursive --preset siglip/v2_so400m_patch14_384 --precision bfloat16 \
    --classes cat dog --threshold 0.1 --format jsonl > results.jsonl
open-vocabulary-classify large_photo.jpg --weights google/siglip2-base-patch16-naflex --backend siglip \
    --tile-grid 3 --classes "small bird" "airplane"
open-vocabulary-classify shelf/ --preset siglip/v2_base_patch16_224 --classes "my mug:mug,@refs/my_mug/" cup
```
