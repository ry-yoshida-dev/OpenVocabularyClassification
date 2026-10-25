from dataclasses import dataclass

from ..array_types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class ClassLogits:
    """
    Class logits of one image and the embedding each class matched best.

    Attributes
    ----------
    logits : FloatArray
        Logit of every class, shape (C,).
    matched_entry_ids : IntArray
        Highest sentence of the highest candidate of every class, indexing the entries of the layout, shape (C,).
    """

    logits: FloatArray
    matched_entry_ids: IntArray
