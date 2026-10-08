from typing import cast

from transformers import AutoConfig, PretrainedConfig


class CheckpointConfig:
    """
    Reads the configuration of a ``transformers`` checkpoint without loading its weights.

    Backends covering several architectures (CLIP and MetaCLIP 2, fixed-resolution SigLIP and SigLIP 2 NaFlex) choose
    their classifier from the ``model_type`` stored in the checkpoint's ``config.json``.
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

        Raises
        ------
        TypeError
            If ``transformers`` does not return a model configuration.
        """
        config: object = cast(object, AutoConfig.from_pretrained(weights_path))
        if not isinstance(config, PretrainedConfig):
            raise TypeError(f"expected a model configuration for {weights_path}. got {type(config)}")
        return config.model_type
