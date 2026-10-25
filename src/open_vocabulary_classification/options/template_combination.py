from enum import StrEnum

from .logit_combination import LogitCombination


class TemplateCombination(StrEnum):
    """
    How the sentences of a text query, one per text template, are combined.

    Attributes
    ----------
    ENSEMBLE : str
        The L2-normalized sentence embeddings are averaged and normalized again into one embedding per text query
        (prompt ensembling, as in the CLIP paper).
    SEPARATE : str
        Every sentence keeps its own embedding and becomes a candidate of its class of its own.
    MAX : str
        Every sentence keeps its own embedding; the highest sentence logit is the query logit.
    MEAN : str
        Every sentence keeps its own embedding; the mean sentence logit is the query logit.
    MEDIAN : str
        Every sentence keeps its own embedding; the median sentence logit is the query logit.
    MIN : str
        Every sentence keeps its own embedding; the lowest sentence logit is the query logit.
    """

    ENSEMBLE = "ensemble"
    SEPARATE = "separate"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"

    @property
    def is_ensembled(self) -> bool:
        """
        Whether the sentences of a text query are averaged into one embedding.

        Returns
        -------
        bool
            ``True`` only for ``ENSEMBLE``.
        """
        return self is TemplateCombination.ENSEMBLE

    @property
    def logit_combination(self) -> LogitCombination | None:
        """
        How the sentence logits of a text query become its logit.

        Returns
        -------
        LogitCombination | None
            Reduction over the sentences of a query; ``None`` when no sentence logits are reduced, because the
            sentences were ensembled into one embedding or every sentence is a candidate of its own.
        """
        match self:
            case TemplateCombination.ENSEMBLE | TemplateCombination.SEPARATE:
                return None
            case TemplateCombination.MAX:
                return LogitCombination.MAX
            case TemplateCombination.MEAN:
                return LogitCombination.MEAN
            case TemplateCombination.MEDIAN:
                return LogitCombination.MEDIAN
            case TemplateCombination.MIN:
                return LogitCombination.MIN
