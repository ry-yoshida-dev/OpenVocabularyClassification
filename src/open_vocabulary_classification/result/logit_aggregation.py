from dataclasses import dataclass, field

import numpy as np

from ..array_types import FloatArray, IntArray
from ..options import AggregationTiming
from ..prompt import EmbeddingEntry, EmbeddingLayout, EntryGroups
from .class_logits import ClassLogits
from .logit_reduction import LogitReduction


@dataclass(frozen=True, eq=False)
class LogitAggregation:
    """
    How the embedding logits of a prompt layout become class logits, prepared once per prompt.

    The ``SIMILARITY`` steps of the layout's aggregation run in order, each reducing its groups of logits; together
    with the ``EMBEDDING`` steps the layout went through, one logit per class remains. Every logit keeps the
    embedding of the highest logit it was reduced from, reported as the match of its class.

    Attributes
    ----------
    layout : EmbeddingLayout
        Embeddings of the prompt that are scored.
    reductions : tuple[LogitReduction, ...]
        Reduction of every ``SIMILARITY`` step, in order.

    Raises
    ------
    ValueError
        If the steps leave other than one logit per class.
    """

    layout: EmbeddingLayout
    reductions: tuple[LogitReduction, ...] = field(init=False)

    def __post_init__(self) -> None:
        entries: list[EmbeddingEntry] = list(self.layout.entries)
        reductions: list[LogitReduction] = []
        for step in self.layout.aggregation.steps_at(AggregationTiming.SIMILARITY):
            groups: EntryGroups = EntryGroups.of(entries, step.unit)
            reductions.append(LogitReduction(group_ids=groups.group_ids, method=step.method))
            entries = list(groups.merged_entries)
        if [entry.class_id for entry in entries] != list(range(len(self.layout.prompt.class_names))):
            raise ValueError("the aggregation steps must leave exactly one logit per class.")
        object.__setattr__(self, "reductions", tuple(reductions))

    def combine(self, embedding_logits: FloatArray) -> ClassLogits:
        """
        Combine the embedding logits of one image into class logits.

        Parameters
        ----------
        embedding_logits : FloatArray
            Logit of every embedding of the layout, shape (E,).

        Returns
        -------
        ClassLogits
            Logit of every class and the embedding of the highest logit it was reduced from.

        Raises
        ------
        ValueError
            If ``embedding_logits`` does not have shape (E,).
        """
        if embedding_logits.shape != (len(self.layout),):
            raise ValueError(f"embedding_logits must have shape ({len(self.layout)},). got {embedding_logits.shape}")
        logits: FloatArray = embedding_logits.astype(np.float64, copy=True)
        best_entry_ids: IntArray = np.arange(logits.size, dtype=np.int64)
        for reduction in self.reductions:
            logits, best_entry_ids = reduction.apply(logits, best_entry_ids)
        return ClassLogits(logits=logits, matched_entry_ids=best_entry_ids)
