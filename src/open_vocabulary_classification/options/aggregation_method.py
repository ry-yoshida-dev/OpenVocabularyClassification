from enum import StrEnum


class AggregationMethod(StrEnum):
    """
    How several values become one in an aggregation step.

    Embeddings are aggregated by ``MEAN`` only: the L2-normalized embeddings are averaged and normalized again, so the
    representative embedding still lies in the joint image-text space. Every sentence counts as one member and every
    visual query as one member averaging its references, so a text query weighs as many members as there are
    templates. Logits take every method.

    Attributes
    ----------
    MEAN : str
        The mean: every value counts equally.
    MAX : str
        The highest logit: one strong match is enough.
    MEDIAN : str
        The median logit: like the mean, but single outliers do not move it.
    MIN : str
        The lowest logit: every logit must be high.
    LOGSUMEXP : str
        ``log(sum(exp(logits)))``, a smooth maximum. With a softmax activation, the score of a class is then the sum
        of the softmax scores its logits would get as classes of their own.
    """

    MEAN = "mean"
    MAX = "max"
    MEDIAN = "median"
    MIN = "min"
    LOGSUMEXP = "logsumexp"
