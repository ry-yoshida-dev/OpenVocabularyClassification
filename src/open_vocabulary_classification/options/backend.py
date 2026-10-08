from enum import StrEnum

from .image_pooling import ImagePooling
from .score_activation import ScoreActivation


class ClassifierBackend(StrEnum):
    """
    Classifier family.

    Model-specific values (weights, pooling, text templates) live in the YAML presets; only what code must know about
    a family, the image poolings its vision encoder offers and how its logits become scores, is defined here.

    Attributes
    ----------
    CLIP : str
        CLIP-architecture models through Hugging Face ``transformers``: OpenAI CLIP, OpenCLIP and MetaCLIP weights
        (``model_type`` ``clip``) and MetaCLIP 2 (``metaclip_2``); the checkpoint decides which.
    SIGLIP : str
        SigLIP and SigLIP 2 through Hugging Face ``transformers``: fixed-resolution weights (``model_type`` ``siglip``)
        and SigLIP 2 NaFlex weights keeping the aspect ratio (``siglip2``); the checkpoint decides which.
    """

    CLIP = "clip"
    SIGLIP = "siglip"

    @property
    def supported_image_poolings(self) -> frozenset[ImagePooling]:
        """
        Poolings the vision encoder of the family offers.

        Returns
        -------
        frozenset[ImagePooling]
            ``CLASS_TOKEN`` and ``PATCH_MEAN`` for CLIP, which has a class token; ``ATTENTION`` and ``PATCH_MEAN`` for
            SigLIP, which has an attention pooling head instead.
        """
        match self:
            case ClassifierBackend.CLIP:
                return frozenset({ImagePooling.CLASS_TOKEN, ImagePooling.PATCH_MEAN})
            case ClassifierBackend.SIGLIP:
                return frozenset({ImagePooling.ATTENTION, ImagePooling.PATCH_MEAN})

    @property
    def score_activation(self) -> ScoreActivation:
        """
        Activation matching the training loss of the family.

        Returns
        -------
        ScoreActivation
            ``SOFTMAX`` for CLIP, ``SIGMOID`` for SigLIP.
        """
        match self:
            case ClassifierBackend.CLIP:
                return ScoreActivation.SOFTMAX
            case ClassifierBackend.SIGLIP:
                return ScoreActivation.SIGMOID
