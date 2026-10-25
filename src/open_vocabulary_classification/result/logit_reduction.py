from dataclasses import dataclass, field

import numpy as np

from ..array_types import FloatArray, IntArray
from ..options import AggregationMethod
from .logit_segments import LogitSegments


@dataclass(frozen=True, eq=False)
class LogitReduction:
    """
    One aggregation step on logits: the logits of every group become one, prepared once per prompt.

    The logits are sorted by group, so every group is a consecutive segment reduced by a vectorized NumPy call.

    Attributes
    ----------
    group_ids : IntArray
        Group of every input logit, numbered from 0 without gaps, shape (K,).
    method : AggregationMethod
        How the logits of a group become one.
    order : IntArray
        Input indices sorted by group, stable, shape (K,).
    segments : LogitSegments
        Groups as segments of the sorted logits.

    Raises
    ------
    ValueError
        If ``group_ids`` is empty or skips a group number.
    """

    group_ids: IntArray
    method: AggregationMethod
    order: IntArray = field(init=False)
    segments: LogitSegments = field(init=False)

    def __post_init__(self) -> None:
        if self.group_ids.size == 0 or np.unique(self.group_ids).tolist() != list(range(int(self.group_ids.max()) + 1)):
            raise ValueError(f"group_ids must number groups from 0 without gaps. got {self.group_ids.tolist()}")
        order: IntArray = np.argsort(self.group_ids, kind="stable").astype(np.int64)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "segments", LogitSegments.of(self.group_ids[order]))

    def apply(self, logits: FloatArray, best_entry_ids: IntArray) -> tuple[FloatArray, IntArray]:
        """
        Reduce the logits of every group and keep the entry of its highest logit.

        Parameters
        ----------
        logits : FloatArray
            Input logits, shape (K,).
        best_entry_ids : IntArray
            Highest embedding behind every input logit, shape (K,).

        Returns
        -------
        tuple[FloatArray, IntArray]
            Logit of every group and the highest embedding behind it, shape (G,) each.
        """
        sorted_logits: FloatArray = logits[self.order]
        reduced_logits: FloatArray = self.segments.combine(sorted_logits, self.method)
        return reduced_logits, best_entry_ids[self.order][self.segments.argmax(sorted_logits)]
