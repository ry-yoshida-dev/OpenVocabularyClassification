# classifier

## Overview

Concrete SigLIP-family classifiers. Each loads its model and processor and runs its vision tower on the preprocessed
images; everything else comes from `SiglipFamilyClassifier` in [core.py](../core.py).

## Components

| Component | Description |
| --------- | ----------- |
| [fixed_resolution.py](./fixed_resolution.py) | `SiglipClassifier`: `SiglipModel`, images resized to a square input; SigLIP and fixed-resolution SigLIP 2. |
| [naflex.py](./naflex.py) | `Siglip2NaFlexClassifier`: `Siglip2Model`, aspect ratio kept, padding masked. |
