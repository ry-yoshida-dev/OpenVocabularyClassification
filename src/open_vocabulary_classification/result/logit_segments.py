from dataclasses import dataclass, field

import numpy as np

from ..array_types import BoolArray, FloatArray, IntArray
from ..options import LogitCombination


@dataclass(frozen=True, eq=False)
class LogitSegments:
    """
    Consecutive, non-empty runs of a logit vector, each combined into one logit by a few vectorized NumPy calls.

    Attributes
    ----------
    starts : IntArray
        First index of every segment, strictly increasing from 0, shape (G,).
    size : int
        Length of the logit vectors the segments cover.
    lengths : IntArray
        Number of logits of every segment, shape (G,).
    segment_ids : IntArray
        Segment of every logit, shape (size,).
    is_singleton : bool
        Whether every segment holds exactly one logit, so combining returns the logits as they are.

    Raises
    ------
    ValueError
        If ``starts`` is empty, does not start at 0, is not strictly increasing or reaches ``size``.
    """

    starts: IntArray
    size: int
    lengths: IntArray = field(init=False)
    segment_ids: IntArray = field(init=False)
    is_singleton: bool = field(init=False)

    def __post_init__(self) -> None:
        is_valid: bool = (
            self.starts.ndim == 1
            and self.starts.size > 0
            and int(self.starts[0]) == 0
            and bool(np.all(np.diff(self.starts) > 0))
            and int(self.starts[-1]) < self.size
        )
        if not is_valid:
            raise ValueError(f"starts must increase strictly from 0 below {self.size}. got {self.starts.tolist()}")
        lengths: IntArray = np.diff(self.starts, append=self.size)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "segment_ids", np.repeat(np.arange(lengths.size, dtype=np.int64), lengths))
        object.__setattr__(self, "is_singleton", lengths.size == self.size)

    @classmethod
    def of(cls, labels: IntArray) -> "LogitSegments":
        """
        Segments of the runs of equal labels.

        Parameters
        ----------
        labels : IntArray
            Label of every logit, equal labels next to each other, shape (K,) with K > 0.

        Returns
        -------
        LogitSegments
            One segment per run of equal labels.
        """
        is_start: BoolArray = np.concatenate(([True], labels[1:] != labels[:-1]))
        return cls(starts=np.flatnonzero(is_start).astype(np.int64), size=labels.size)

    def combine(self, logits: FloatArray, combination: LogitCombination) -> FloatArray:
        """
        Combine the logits of every segment into one.

        Parameters
        ----------
        logits : FloatArray
            Logits the segments cover, shape (size,).
        combination : LogitCombination
            How the logits of a segment become one.

        Returns
        -------
        FloatArray
            Combined logit of every segment, shape (G,).

        Raises
        ------
        ValueError
            If ``logits`` does not have shape (size,).
        """
        self._validate(logits)
        if self.is_singleton:
            return logits.astype(np.float64, copy=True)
        match combination:
            case LogitCombination.MAX:
                return np.maximum.reduceat(logits, self.starts)
            case LogitCombination.MEAN:
                return np.add.reduceat(logits, self.starts) / self.lengths
            case LogitCombination.MEDIAN:
                return self._medians(logits)
            case LogitCombination.MIN:
                return np.minimum.reduceat(logits, self.starts)

    def argmax(self, logits: FloatArray) -> IntArray:
        """
        Index of the highest logit of every segment.

        Parameters
        ----------
        logits : FloatArray
            Logits the segments cover, shape (size,).

        Returns
        -------
        IntArray
            Index into ``logits`` of the highest logit of every segment, the first one on ties, shape (G,).

        Raises
        ------
        ValueError
            If ``logits`` does not have shape (size,).
        """
        self._validate(logits)
        indices: IntArray = np.arange(self.size, dtype=np.int64)
        if self.is_singleton:
            return indices
        segment_maxima: FloatArray = np.maximum.reduceat(logits, self.starts)
        maximum_indices: IntArray = np.where(logits == segment_maxima[self.segment_ids], indices, self.size)
        return np.minimum.reduceat(maximum_indices, self.starts)

    def _medians(self, logits: FloatArray) -> FloatArray:
        medians: FloatArray = np.empty(self.lengths.size, dtype=np.float64)
        for length in np.unique(self.lengths):
            segments: IntArray = np.flatnonzero(self.lengths == length)
            member_indices: IntArray = self.starts[segments, np.newaxis] + np.arange(length, dtype=np.int64)
            medians[segments] = np.median(logits[member_indices], axis=1)
        return medians

    def _validate(self, logits: FloatArray) -> None:
        if logits.shape != (self.size,):
            raise ValueError(f"logits must have shape ({self.size},). got {logits.shape}")
