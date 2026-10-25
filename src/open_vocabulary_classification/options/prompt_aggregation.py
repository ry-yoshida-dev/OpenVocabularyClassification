from collections.abc import Sequence
from dataclasses import dataclass

from .aggregation_method import AggregationMethod
from .aggregation_step import AggregationStep
from .aggregation_timing import AggregationTiming
from .aggregation_unit import AggregationUnit


@dataclass(frozen=True)
class PromptAggregation:
    """
    The aggregation steps turning the embeddings of every class, one per prompt and template, into one class logit.

    Either one ``TEMPLATE_AND_PROMPT`` step merges everything at once, or a ``TEMPLATE`` and a ``PROMPT`` step merge
    the two axes one after the other, in the given order; a later step never runs at an earlier timing. At most one
    step runs at ``EMBEDDING``: it averages every member of its group at once, so stacking two would equal one
    ``TEMPLATE_AND_PROMPT`` step instead of a mean of means. The default is prompt ensembling as in the CLIP paper
    followed by the best prompt of each class.

    Attributes
    ----------
    steps : tuple[AggregationStep, ...]
        Steps in the order they run.

    Raises
    ------
    ValueError
        If the steps do not merge both axes exactly once, run at a decreasing timing or more than one runs at
        ``EMBEDDING``.
    """

    steps: tuple[AggregationStep, ...] = (
        AggregationStep(AggregationUnit.TEMPLATE, AggregationTiming.EMBEDDING, AggregationMethod.MEAN),
        AggregationStep(AggregationUnit.PROMPT, AggregationTiming.SIMILARITY, AggregationMethod.MAX),
    )

    def __post_init__(self) -> None:
        units: list[AggregationUnit] = [step.unit for step in self.steps]
        is_complete: bool = units == [AggregationUnit.TEMPLATE_AND_PROMPT] or sorted(units) == sorted(
            [AggregationUnit.TEMPLATE, AggregationUnit.PROMPT]
        )
        if not is_complete:
            raise ValueError(
                "aggregation needs one template_and_prompt step, or one template and one prompt step. "
                + f"got {[unit.value for unit in units]}"
            )
        timings: list[AggregationTiming] = [step.timing for step in self.steps]
        order: list[AggregationTiming] = list(AggregationTiming)
        if [order.index(timing) for timing in timings] != sorted(order.index(timing) for timing in timings):
            raise ValueError(
                f"a later step must not run at an earlier timing. got {[timing.value for timing in timings]}"
            )
        if timings.count(AggregationTiming.EMBEDDING) > 1:
            raise ValueError(
                "at most one step runs at embedding; use one template_and_prompt step to average the whole table."
            )

    @classmethod
    def of(cls, steps: Sequence[AggregationStep]) -> "PromptAggregation":
        """
        Build an aggregation of the given steps.

        Parameters
        ----------
        steps : Sequence[AggregationStep]
            Steps in the order they run.

        Returns
        -------
        PromptAggregation
            Validated aggregation.
        """
        return cls(steps=tuple(steps))

    def steps_at(self, timing: AggregationTiming) -> tuple[AggregationStep, ...]:
        """
        Select the steps running at one timing, in order.

        Parameters
        ----------
        timing : AggregationTiming
            Timing to select.

        Returns
        -------
        tuple[AggregationStep, ...]
            Steps with that timing.
        """
        return tuple(step for step in self.steps if step.timing is timing)
