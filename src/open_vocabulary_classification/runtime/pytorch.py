import torch

from ..options import Device
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
            If an explicitly requested GPU backend is unavailable.
        ValueError
            If half precision is requested and the device resolves to the CPU.
        """
        self._device: torch.device = self._resolve_device(settings.device)
        if settings.is_half_precision_enabled and self._device.type == Device.CPU:
            raise ValueError("half precision is not supported on the CPU.")
        self._dtype: torch.dtype = torch.float16 if settings.is_half_precision_enabled else torch.float32

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
            ``torch.float16`` with half precision, otherwise ``torch.float32``.
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
