# classifier

## Overview

Concrete CLIP-family classifiers. Each loads its model class; everything else comes from `ClipFamilyClassifier` in
[core.py](../core.py).

## Components

| Component | Description |
| --------- | ----------- |
| [clip.py](./clip.py) | `ClipClassifier`: `CLIPModel`, for OpenAI CLIP, OpenCLIP and MetaCLIP weights. |
| [metaclip2.py](./metaclip2.py) | `MetaClip2Classifier`: `MetaClip2Model`, multilingual MetaCLIP 2. |
