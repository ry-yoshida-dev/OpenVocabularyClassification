from enum import StrEnum


class LogitCombination(StrEnum):
    """
    How several logits become one, e.g. the sentences of a text query or the queries of a class.

    Attributes
    ----------
    MAX : str
        The highest logit: one strong match is enough.
    MEAN : str
        The mean logit: every logit counts equally.
    MEDIAN : str
        The median logit: like the mean, but single outliers do not move it.
    MIN : str
        The lowest logit: every logit must be high.
    """

    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
