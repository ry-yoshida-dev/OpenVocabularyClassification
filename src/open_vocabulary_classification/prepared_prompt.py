from dataclasses import dataclass

from .array_types import FloatArray
from .prompt import EmbeddingLayout
from .result import LogitAggregation


@dataclass(frozen=True, eq=False)
class PreparedPrompt:
    """
    Embeddings and logit aggregation of a prompt, prepared once and reused for every image a classifier scores.

    Attributes
    ----------
    layout : EmbeddingLayout
        Embeddings of the prompt.
    aggregation : LogitAggregation
        How the logits of the layout are combined per class.
    embeddings : FloatArray
        Read-only L2-normalized embeddings in embedding-id order, shape (E, D).
    """

    layout: EmbeddingLayout
    aggregation: LogitAggregation
    embeddings: FloatArray
