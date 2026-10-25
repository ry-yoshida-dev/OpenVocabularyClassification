from collections.abc import Mapping
from dataclasses import dataclass

from .aggregation_method import AggregationMethod
from .aggregation_timing import AggregationTiming
from .aggregation_unit import AggregationUnit


@dataclass(frozen=True)
class AggregationStep:
    """
    One aggregation of the embeddings of every class: which embeddings are merged, where and how.

    Attributes
    ----------
    unit : AggregationUnit
        Which embeddings of a class are merged.
    timing : AggregationTiming
        Whether the embeddings or their logits are merged.
    method : AggregationMethod
        How the merged values become one; ``MEAN`` for ``EMBEDDING``.

    Raises
    ------
    ValueError
        If embeddings are to be aggregated by another method than ``MEAN``.
    """

    unit: AggregationUnit
    timing: AggregationTiming
    method: AggregationMethod

    def __post_init__(self) -> None:
        if self.timing is AggregationTiming.EMBEDDING and self.method is not AggregationMethod.MEAN:
            raise ValueError(f"embeddings are aggregated by {AggregationMethod.MEAN} only. got {self.method}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "AggregationStep":
        """
        Build a step from plain values, e.g. ``{"unit": "template", "timing": "embedding", "method": "mean"}``.

        Parameters
        ----------
        values : Mapping[str, object]
            ``unit``, ``timing`` and ``method`` by value.

        Returns
        -------
        AggregationStep
            Validated step.

        Raises
        ------
        KeyError
            If a key is missing or unknown.
        ValueError
            If a value is unknown or the step is invalid.
        """
        known_keys: frozenset[str] = frozenset({"unit", "timing", "method"})
        if frozenset(values) != known_keys:
            raise KeyError(f"an aggregation step needs exactly the keys {sorted(known_keys)}. got {sorted(values)}")
        return cls(
            unit=AggregationUnit(str(values["unit"])),
            timing=AggregationTiming(str(values["timing"])),
            method=AggregationMethod(str(values["method"])),
        )

    def to_mapping(self) -> dict[str, str]:
        """
        Plain values of the step, the inverse of ``from_mapping``.

        Returns
        -------
        dict[str, str]
            ``unit``, ``timing`` and ``method`` by value.
        """
        return {"unit": self.unit.value, "timing": self.timing.value, "method": self.method.value}
