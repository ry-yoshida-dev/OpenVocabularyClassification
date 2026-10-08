from enum import StrEnum

from ...classifier import OpenVocabularyClassifier
from ...settings import ClassifierSettings
from ..checkpoint_config import CheckpointConfig
from .classifier import Siglip2NaFlexClassifier, SiglipClassifier


class SiglipArchitecture(StrEnum):
    """
    SigLIP-family architecture, identified by the ``model_type`` of a checkpoint.

    Attributes
    ----------
    FIXED_RESOLUTION : str
        SigLIP and the fixed-resolution SigLIP 2 weights, e.g. ``google/siglip2-base-patch16-224``.
    NAFLEX : str
        SigLIP 2 NaFlex weights keeping the aspect ratio, e.g. ``google/siglip2-base-patch16-naflex``.
    """

    FIXED_RESOLUTION = "siglip"
    NAFLEX = "siglip2"

    @classmethod
    def of_checkpoint(cls, weights_path: str) -> "SiglipArchitecture":
        """
        Read the architecture of a checkpoint from its configuration.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        SiglipArchitecture
            Architecture of the checkpoint.

        Raises
        ------
        ValueError
            If the checkpoint is not a SigLIP-family model.
        """
        model_type: str = CheckpointConfig.read_model_type(weights_path)
        model_types: list[str] = sorted(architecture.value for architecture in cls)
        if model_type not in model_types:
            raise ValueError(f"{weights_path} is a {model_type!r} model, not one of {model_types}.")
        return cls(model_type)

    def build(self, settings: ClassifierSettings) -> OpenVocabularyClassifier:
        """
        Load the classifier of this architecture.

        Parameters
        ----------
        settings : ClassifierSettings
            Checkpoint of this architecture, pooling, text templates, batching and device.

        Returns
        -------
        OpenVocabularyClassifier
            Loaded ``SiglipClassifier`` or ``Siglip2NaFlexClassifier``.
        """
        match self:
            case SiglipArchitecture.FIXED_RESOLUTION:
                return SiglipClassifier(settings)
            case SiglipArchitecture.NAFLEX:
                return Siglip2NaFlexClassifier(settings)
