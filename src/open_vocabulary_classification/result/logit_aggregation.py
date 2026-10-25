from dataclasses import dataclass, field

import numpy as np

from ..array_types import FloatArray, IntArray
from ..options import AggregationTiming, PromptAggregation
from ..prompt import EmbeddingLayout, EntryGroups, EntryLabel
from .candidate_logits import CandidateLogits
from .logit_reduction import LogitReduction
from .logit_segments import LogitSegments


@dataclass(frozen=True, eq=False)
class LogitAggregation:
    """
    How the embedding logits of a prompt become candidate logits, prepared once per prompt.

    The ``SIMILARITY`` steps of the aggregation run in order, each reducing its groups of logits. What remains are
    the candidates of every class, one per class when both axes are merged, ordered by class. Every candidate keeps
    the embedding of the highest logit it was reduced from, reported as the match of its class. Only the labels of
    the embeddings are read, so logits stored with their labels can be aggregated again with other ``SIMILARITY``
    steps; the ``EMBEDDING`` steps of ``aggregation`` are not read.

    Attributes
    ----------
    entry_labels : tuple[EntryLabel, ...]
        Label of every embedding, ordered by class.
    aggregation : PromptAggregation
        Aggregation whose ``SIMILARITY`` steps reduce the logits.
    reductions : tuple[LogitReduction, ...]
        Reduction of every ``SIMILARITY`` step, in order.
    class_segments : LogitSegments
        Candidates of every class.

    Raises
    ------
    ValueError
        If there is no embedding or the embeddings are not ordered by class.
    """

    entry_labels: tuple[EntryLabel, ...]
    aggregation: PromptAggregation
    reductions: tuple[LogitReduction, ...] = field(init=False)
    class_segments: LogitSegments = field(init=False)

    def __post_init__(self) -> None:
        class_ids: list[int] = [label.class_id for label in self.entry_labels]
        if not class_ids or class_ids != sorted(class_ids):
            raise ValueError("entry_labels must hold at least one embedding, ordered by class.")
        labels: tuple[EntryLabel, ...] = self.entry_labels
        reductions: list[LogitReduction] = []
        for step in self.aggregation.steps_at(AggregationTiming.SIMILARITY):
            groups: EntryGroups = EntryGroups.of(labels, step.unit)
            reductions.append(LogitReduction(group_ids=groups.group_ids, method=step.method))
            labels = groups.labels
        object.__setattr__(self, "reductions", tuple(reductions))
        candidate_class_ids: IntArray = np.array([label.class_id for label in labels], dtype=np.int64)
        object.__setattr__(self, "class_segments", LogitSegments.of(candidate_class_ids))

    @classmethod
    def of(cls, layout: EmbeddingLayout) -> "LogitAggregation":
        """
        Prepare the aggregation of a prompt layout.

        Parameters
        ----------
        layout : EmbeddingLayout
            Embeddings of the prompt that are scored.

        Returns
        -------
        LogitAggregation
            Reductions of the ``SIMILARITY`` steps of the layout's aggregation.
        """
        return cls(entry_labels=layout.entry_labels, aggregation=layout.aggregation)

    def __len__(self) -> int:
        return len(self.entry_labels)

    def combine(self, embedding_logits: FloatArray) -> CandidateLogits:
        """
        Reduce the embedding logits of one image to the logits of the candidates.

        Parameters
        ----------
        embedding_logits : FloatArray
            Logit of every embedding, shape (E,).

        Returns
        -------
        CandidateLogits
            Logit of every candidate and the embedding of the highest logit it was reduced from.

        Raises
        ------
        ValueError
            If ``embedding_logits`` does not have shape (E,).
        """
        if embedding_logits.shape != (len(self),):
            raise ValueError(f"embedding_logits must have shape ({len(self)},). got {embedding_logits.shape}")
        logits: FloatArray = embedding_logits.astype(np.float64, copy=True)
        best_entry_ids: IntArray = np.arange(logits.size, dtype=np.int64)
        for reduction in self.reductions:
            logits, best_entry_ids = reduction.apply(logits, best_entry_ids)
        return CandidateLogits(logits=logits, best_entry_ids=best_entry_ids)
