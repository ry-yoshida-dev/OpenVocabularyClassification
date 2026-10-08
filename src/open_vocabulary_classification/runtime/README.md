# runtime

## Overview

How a classifier's model is executed. `OpenVocabularyClassifier` itself is independent of any inference framework;
each classifier holds the runtime its model needs. Every current backend runs on PyTorch through `TorchRuntime`; an
ONNX Runtime or TensorRT runtime would be added here and hand its logits and embeddings to the shared
post-processing as NumPy arrays.

## Components

| Component | Description |
| --------- | ----------- |
| [pytorch.py](./pytorch.py) | `TorchRuntime`: resolves `Device` to a `torch.device` and `Precision` to a dtype (rejecting `float16` on the CPU and unsupported `bfloat16` on CUDA), and prepares models and inputs. |

## Examples

```python
runtime = TorchRuntime(settings)
runtime.prepare_model(model)
pixel_values = runtime.to_model_input(pixel_values)
input_ids = runtime.to_device(input_ids)
```
