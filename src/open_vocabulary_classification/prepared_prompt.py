from dataclasses import dataclass

from .array_types import FloatArray
from .result import LogitAggregation


@dataclass(frozen=True, eq=False)
class PreparedPrompt:
    """
    Embeddings and logit aggregation of a prompt, prepared once and reused for every image a classifier scores.

    Attributes
    ----------
    aggregation : LogitAggregation
        Layout of the prompt and how its logits are combined per class.
    embeddings : FloatArray
        Read-only L2-normalized embeddings in embedding-id order, shape (E, D).
    """

    aggregation: LogitAggregation
    embeddings: FloatArray
