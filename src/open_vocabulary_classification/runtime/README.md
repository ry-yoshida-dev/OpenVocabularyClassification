# runtime

## Overview

How a classifier's model is executed. Every current backend runs on PyTorch through `TorchRuntime`.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`, which runs PyTorch models on the requested device and precision. |

## Examples

```python
runtime = TorchRuntime(settings)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
input_ids = runtime.to_device(input_ids)
```
