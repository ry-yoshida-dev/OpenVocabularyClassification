import torch

from ..options import Device, Precision
from ..settings import ClassifierSettings


class TorchRuntime:
    """
    Runs PyTorch models on the device and precision requested by ``ClassifierSettings``.

    Classifiers backed by PyTorch hold one; classifiers backed by another runtime (ONNX Runtime, TensorRT)
    would hold their own runtime instead, while sharing everything else through ``OpenVocabularyClassifier``.
    """

    def __init__(self, settings: ClassifierSettings) -> None:
        """
        Parameters
        ----------
        settings : ClassifierSettings
            Device and precision to run with.

        Raises
        ------
        RuntimeError
            If an explicitly requested GPU backend, or bfloat16 on CUDA, is unavailable.
        ValueError
            If float16 is requested and the device resolves to the CPU.
        """
        self._device: torch.device = self._resolve_device(settings.device)
        self._dtype: torch.dtype = self._resolve_dtype(settings.precision, self._device)

    @property
    def device(self) -> torch.device:
        """
        Device the model runs on.

        Returns
        -------
        torch.device
            Resolved device.
        """
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """
        Floating-point type of the model weights and image inputs.

        Returns
        -------
        torch.dtype
            ``torch.float32``, ``torch.float16`` or ``torch.bfloat16`` as ``settings.precision`` says.
        """
        return self._dtype

    def prepare_model(self, model: torch.nn.Module) -> None:
        """
        Put a model in evaluation mode on the runtime device and dtype.

        Parameters
        ----------
        model : torch.nn.Module
            Model to prepare in place.
        """
        model.eval()
        model.to(device=self._device, dtype=self._dtype)

    def to_model_input(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Move image pixels to the runtime device and dtype.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Preprocessed image batch.

        Returns
        -------
        torch.Tensor
            Pixels ready for the model.
        """
        return pixel_values.to(device=self._device, dtype=self._dtype)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move a non-pixel model input, e.g. a token id or mask tensor, to the runtime device keeping its dtype.

        Parameters
        ----------
        tensor : torch.Tensor
            Model input.

        Returns
        -------
        torch.Tensor
            Input on the runtime device.
        """
        return tensor.to(device=self._device)

    @staticmethod
    def _resolve_device(device: Device) -> torch.device:
        match device:
            case Device.AUTO:
                if torch.cuda.is_available():
                    return torch.device(Device.CUDA)
                if torch.backends.mps.is_available():
                    return torch.device(Device.MPS)
                return torch.device(Device.CPU)
            case Device.CPU:
                return torch.device(Device.CPU)
            case Device.CUDA:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is not available.")
                return torch.device(Device.CUDA)
            case Device.MPS:
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS was requested but is not available.")
                return torch.device(Device.MPS)

    @staticmethod
    def _resolve_dtype(precision: Precision, device: torch.device) -> torch.dtype:
        match precision:
            case Precision.FLOAT32:
                return torch.float32
            case Precision.FLOAT16:
                if device.type == Device.CPU:
                    raise ValueError("float16 is not supported on the CPU; use bfloat16 or float32.")
                return torch.float16
            case Precision.BFLOAT16:
                if device.type == Device.CUDA and not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bfloat16 was requested but the CUDA device does not support it.")
                return torch.bfloat16
