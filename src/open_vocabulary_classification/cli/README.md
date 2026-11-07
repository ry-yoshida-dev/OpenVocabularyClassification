# cli

## Overview

The `open-vocabulary-classify` command, installed with the package. It classifies image files and directories into classes given as text phrases or reference images (`@path`, an image file or every image of a directory), with settings from a packaged preset, a YAML file or a backend and weights, and prints the top classes of each image as text or JSON lines. Run it with `--help` for every option.

## Components

| Component | Description |
| --------- | ----------- |
| [command.py](./command.py) | `ClassifyCommand`, the command itself. |
| [image_files.py](./image_files.py) | `ImageFiles`, which expands files and directories into image files. |
| [image_file_stream.py](./image_file_stream.py) | `ImageFileStream`, which reads image files lazily. |
| [output_format.py](./output_format.py) | `OutputFormat`: `text` or `jsonl`. |

## Examples

```bash
open-vocabulary-classify --list-presets
open-vocabulary-classify images/ --preset clip/openai_vit_b16 --classes cat "dog:dog,puppy" bus
open-vocabulary-classify images/ --recursive --preset siglip/v2_so400m_patch14_384 --precision bfloat16 \
    --classes cat dog --threshold 0.1 --format jsonl > results.jsonl
open-vocabulary-classify shelf/ --preset siglip/v2_base_patch16_224 --classes "my mug:mug,@refs/my_mug/" cup
```
