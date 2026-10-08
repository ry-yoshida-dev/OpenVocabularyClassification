from enum import StrEnum

from ...classifier import OpenVocabularyClassifier
from ...settings import ClassifierSettings
from ..checkpoint_config import CheckpointConfig
from .classifier import ClipClassifier, MetaClip2Classifier


class ClipArchitecture(StrEnum):
    """
    CLIP-family architecture, identified by the ``model_type`` of a checkpoint.

    Attributes
    ----------
    CLIP : str
        CLIP: OpenAI CLIP, OpenCLIP and MetaCLIP weights.
    METACLIP_2 : str
        MetaCLIP 2.
    """

    CLIP = "clip"
    METACLIP_2 = "metaclip_2"

    @classmethod
    def of_checkpoint(cls, weights_path: str) -> "ClipArchitecture":
        """
        Read the architecture of a checkpoint from its configuration.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        ClipArchitecture
            Architecture of the checkpoint.

        Raises
        ------
        ValueError
            If the checkpoint is not a CLIP-family model.
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
            Loaded ``ClipClassifier`` or ``MetaClip2Classifier``.
        """
        match self:
            case ClipArchitecture.CLIP:
                return ClipClassifier(settings)
            case ClipArchitecture.METACLIP_2:
                return MetaClip2Classifier(settings)
