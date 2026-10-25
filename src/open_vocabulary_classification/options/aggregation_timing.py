from enum import StrEnum


class AggregationTiming(StrEnum):
    """
    Where in the pipeline an aggregation step merges the embeddings of a class.

    Attributes
    ----------
    EMBEDDING : str
        On the text embeddings, before they are compared with the image: the merged embeddings become one
        representative embedding, so the image is compared with fewer embeddings.
    SIMILARITY : str
        On the logits of the embeddings, after they are compared with the image and before the score activation.
    """

    EMBEDDING = "embedding"
    SIMILARITY = "similarity"
