from dataclasses import dataclass

from ..array_types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class CandidateLogits:
    """
    Logits of the candidates left by the aggregation steps for one image, ordered by class.

    Attributes
    ----------
    logits : FloatArray
        Logit of every candidate, shape (K,).
    best_entry_ids : IntArray
        Embedding of the highest logit every candidate was reduced from, indexing the entries of the layout, shape (K,).
    """

    logits: FloatArray
    best_entry_ids: IntArray
