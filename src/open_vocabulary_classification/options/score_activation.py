from enum import StrEnum

import numpy as np

from ..array_types import FloatArray


class ScoreActivation(StrEnum):
    """
    How the logits of a classifier turn into class scores in ``[0, 1]``, fixed by the loss its model was trained with.

    Attributes
    ----------
    SOFTMAX : str
        Softmax over the classes of the prompt (CLIP contrastive loss): scores sum to 1, so a score is relative to the
        other classes and the image is assumed to show exactly one of them.
    SIGMOID : str
        Sigmoid of every class on its own (SigLIP pairwise loss): scores are independent probabilities, so any number
        of classes, or none, may score high.
    """

    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"

    def apply(self, logits: FloatArray) -> FloatArray:
        """
        Turn class logits into scores.

        Parameters
        ----------
        logits : FloatArray
            Logit of every class, classes on the last axis.

        Returns
        -------
        FloatArray
            Scores in ``[0, 1]``, same shape as ``logits``.
        """
        match self:
            case ScoreActivation.SOFTMAX:
                shifted: FloatArray = np.exp(logits - logits.max(axis=-1, keepdims=True))
                return shifted / shifted.sum(axis=-1, keepdims=True)
            case ScoreActivation.SIGMOID:
                return 0.5 * (1.0 + np.tanh(0.5 * logits))
