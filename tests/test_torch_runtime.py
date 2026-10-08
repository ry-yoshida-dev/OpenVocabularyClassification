import pytest
import torch

from open_vocabulary_classification import ClassifierBackend, ClassifierSettings, Device, ImagePooling
from open_vocabulary_classification.runtime import TorchRuntime


def build_settings(device: Device, is_half_precision_enabled: bool = False) -> ClassifierSettings:
    return ClassifierSettings(
        backend=ClassifierBackend.CLIP,
        weights_path="stub",
        image_pooling=ImagePooling.CLASS_TOKEN,
        device=device,
        is_half_precision_enabled=is_half_precision_enabled,
    )


def test_cpu_runtime_prepares_models_and_inputs() -> None:
    runtime: TorchRuntime = TorchRuntime(build_settings(Device.CPU))
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32
    model: torch.nn.Linear = torch.nn.Linear(2, 2).double()
    runtime.prepare_model(model)
    assert not model.training
    assert model.weight.dtype == torch.float32
    assert runtime.to_model_input(torch.zeros(1, 2, dtype=torch.float64)).dtype == torch.float32


def test_auto_device_resolves_to_an_available_device() -> None:
    runtime: TorchRuntime = TorchRuntime(build_settings(Device.AUTO))
    assert runtime.device.type in {Device.CPU, Device.CUDA, Device.MPS}


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_unavailable_cuda_raises() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        TorchRuntime(build_settings(Device.CUDA))


def test_half_precision_on_cpu_raises() -> None:
    with pytest.raises(ValueError, match="half precision"):
        TorchRuntime(build_settings(Device.CPU, is_half_precision_enabled=True))


def test_non_pixel_inputs_keep_their_dtype() -> None:
    runtime: TorchRuntime = TorchRuntime(build_settings(Device.CPU))
    assert runtime.to_device(torch.zeros(2, dtype=torch.int64)).dtype == torch.int64
