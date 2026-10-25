from collections.abc import Sequence
from dataclasses import dataclass

from .aggregation_method import AggregationMethod
from .aggregation_step import AggregationStep
from .aggregation_timing import AggregationTiming
from .aggregation_unit import AggregationUnit


@dataclass(frozen=True)
class PromptAggregation:
    """
    The aggregation steps merging the embeddings of every class, a table of prompts and templates, into candidates.

    Steps run in the given order: steps at ``EMBEDDING`` first, each averaging its groups of embeddings into
    normalized representative embeddings, then steps at ``SIMILARITY``, each reducing the logits of its groups.
    Whatever the steps leave are the candidates of a class, e.g. every sentence without any step. The score
    activation runs over every candidate, and a class scores its best candidate. The default is prompt ensembling as
    in the CLIP paper followed by the best prompt of each class.

    Attributes
    ----------
    steps : tuple[AggregationStep, ...]
        Steps in the order they run.

    Raises
    ------
    ValueError
        If an ``EMBEDDING`` step follows a ``SIMILARITY`` step.
    """

    steps: tuple[AggregationStep, ...] = (
        AggregationStep(AggregationUnit.TEMPLATE, AggregationTiming.EMBEDDING, AggregationMethod.MEAN),
        AggregationStep(AggregationUnit.PROMPT, AggregationTiming.SIMILARITY, AggregationMethod.MAX),
    )

    def __post_init__(self) -> None:
        timings: list[AggregationTiming] = [step.timing for step in self.steps]
        timing_order: list[AggregationTiming] = list(AggregationTiming)
        timing_ranks: list[int] = [timing_order.index(timing) for timing in timings]
        if timing_ranks != sorted(timing_ranks):
            raise ValueError(
                f"embedding steps must not follow similarity steps. got {[timing.value for timing in timings]}"
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
