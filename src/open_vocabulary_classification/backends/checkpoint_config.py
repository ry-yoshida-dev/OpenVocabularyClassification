from typing import cast

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import IMAGE_PROCESSOR_NAME, hub


class CheckpointConfig:
    """
    Reads the configuration of a ``transformers`` checkpoint without loading its weights.

    Backends covering several architectures (CLIP and MetaCLIP 2, fixed-resolution SigLIP and SigLIP 2 NaFlex) choose
    their classifier from the ``model_type`` stored in the checkpoint's ``config.json``. Checkpoints shipped without an
    image processor configuration (e.g. most Apple DFN CLIP weights) get one rebuilt from their input resolution.
    """

    @staticmethod
    def read_model_type(weights_path: str) -> str:
        """
        Read the architecture identifier of a checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        str
            ``model_type`` of the checkpoint, e.g. ``"clip"`` or ``"siglip2"``.
        """
        return CheckpointConfig._read_config(weights_path).model_type

    @staticmethod
    def read_image_size(weights_path: str) -> int:
        """
        Read the input resolution of the vision tower of a checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        int
            Side of the square images the vision tower was trained on, e.g. ``224``.

        Raises
        ------
        TypeError
            If the checkpoint has no vision configuration with an integer ``image_size``.
        """
        config: PretrainedConfig = CheckpointConfig._read_config(weights_path)
        vision_config: object = cast(object, getattr(config, "vision_config", None))
        image_size: object = cast(object, getattr(vision_config, "image_size", None))
        if not isinstance(image_size, int):
            raise TypeError(f"{weights_path} has no integer vision_config.image_size. got {image_size!r}")
        return image_size

    @staticmethod
    def has_image_processor_config(weights_path: str) -> bool:
        """
        Tell whether a checkpoint ships the configuration of its image processor.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        bool
            Whether ``preprocessor_config.json`` is present.
        """
        config_path: str | None = hub.cached_file(
            weights_path, IMAGE_PROCESSOR_NAME, _raise_exceptions_for_missing_entries=False
        )
        return config_path is not None

    @staticmethod
    def _read_config(weights_path: str) -> PretrainedConfig:
        """
        Read the model configuration of a checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        PretrainedConfig
            Configuration stored in the checkpoint's ``config.json``.

        Raises
        ------
        TypeError
            If ``transformers`` does not return a model configuration.
        """
        config: object = cast(object, AutoConfig.from_pretrained(weights_path))
        if not isinstance(config, PretrainedConfig):
            raise TypeError(f"expected a model configuration for {weights_path}. got {type(config)}")
        return config
