from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from ..array_types import FloatArray, IntArray
from ..options import ScoreActivation
from ..prompt import EmbeddingEntry, EmbeddingLayout, Prompt
from .class_scores import ClassScores
from .classification import Classification
from .classification_record import ClassificationRecord
from .logit_aggregation import LogitAggregation


@dataclass(frozen=True, eq=False)
class ClassificationResult:
    """
    Model-independent class scores for one image.

    ``aggregation`` reduces the logits of the embeddings of every class to its candidates, the activation of the
    classifier turns the logits of every candidate into scores (a softmax shares the score among the candidates of
    every class), and each class takes its best candidate: its score and logit, as ``ClassScores`` computes them. With
    one candidate per class this is the activation over the class logits. The matched embedding of a class is the one
    with the highest logit among those its logit was reduced from; its query and template are reported as matched,
    e.g. ``"puppy"`` for class ``"dog"``, where the embedding still has a single one. With a method other than ``max``
    the class logit is not the logit of the matched embedding.
    Iterating yields every class in class-id order. The arrays are read-only; ``embedding_logits`` is a copy of the
    given logits.

    Attributes
    ----------
    layout : EmbeddingLayout
        Embeddings of the prompt that were scored.
    aggregation : LogitAggregation
        How the logits of the layout are combined per class.
    embedding_logits : FloatArray
        Logit of every embedding, shape (E,).
    score_activation : ScoreActivation
        Activation turning class logits into scores.
    class_logits : FloatArray
        Logit of every class, shape (C,).
    matched_entry_ids : IntArray
        Matched embedding of every class, indexing ``layout.entries``, shape (C,).
    scores : FloatArray
        Score of every class in ``[0, 1]``, shape (C,).

    Raises
    ------
    ValueError
        If ``aggregation`` does not hold the labels of the layout, or ``embedding_logits`` does not hold one finite
        logit per embedding of the layout.
    """

    layout: EmbeddingLayout
    aggregation: LogitAggregation
    embedding_logits: FloatArray
    score_activation: ScoreActivation
    class_logits: FloatArray = field(init=False)
    matched_entry_ids: IntArray = field(init=False)
    scores: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        if self.aggregation.entry_labels != self.layout.entry_labels:
            raise ValueError("aggregation must hold the entry labels of the layout.")
        expected_shape: tuple[int] = (len(self.layout),)
        if self.embedding_logits.shape != expected_shape:
            raise ValueError(f"embedding_logits must have shape {expected_shape}. got {self.embedding_logits.shape}")
        if not bool(np.isfinite(self.embedding_logits).all()):
            raise ValueError("embedding_logits must be finite.")
        embedding_logits: FloatArray = self.embedding_logits.astype(np.float64, copy=True)
        embedding_logits.setflags(write=False)
        class_scores: ClassScores = ClassScores.of(self.aggregation, embedding_logits, self.score_activation)
        object.__setattr__(self, "embedding_logits", embedding_logits)
        object.__setattr__(self, "matched_entry_ids", class_scores.matched_entry_ids)
        object.__setattr__(self, "class_logits", class_scores.class_logits)
        object.__setattr__(self, "scores", class_scores.scores)

    def __len__(self) -> int:
        return len(self.prompt.class_names)

    @property
    def prompt(self) -> Prompt:
        """
        Prompt whose classes were scored.

        Returns
        -------
        Prompt
            Prompt of ``layout``.
        """
        return self.layout.prompt

    def __iter__(self) -> Iterator[Classification]:
        for class_id in range(len(self)):
            yield self.classification(class_id)

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Class names of the prompt.

        Returns
        -------
        tuple[str, ...]
            Class names in class-id order.
        """
        return self.prompt.class_names

    @property
    def top(self) -> Classification:
        """
        Highest-scoring class.

        Returns
        -------
        Classification
            Class with the highest score; the lowest class id on ties.
        """
        return self.classification(int(self._ranked_class_ids()[0]))

    def classification(self, class_id: int) -> Classification:
        """
        Score of one class.

        Parameters
        ----------
        class_id : int
            Index into ``class_names``.

        Returns
        -------
        Classification
            Score, logit, matched query and matched template of the class.

        Raises
        ------
        IndexError
            If ``class_id`` is outside the prompt classes.
        """
        if not 0 <= class_id < len(self):
            raise IndexError(f"class_id must be in [0, {len(self)}). got {class_id}")
        matched_entry: EmbeddingEntry = self.layout.entries[int(self.matched_entry_ids[class_id])]
        return Classification(
            class_id=class_id,
            class_name=self.prompt.class_names[class_id],
            score=float(self.scores[class_id]),
            logit=float(self.class_logits[class_id]),
            matched_query=None if matched_entry.query_id is None else self.prompt.queries[matched_entry.query_id],
            matched_template=matched_entry.template,
        )

    def score_of(self, class_name: str) -> float:
        """
        Score of a class looked up by name.

        Parameters
        ----------
        class_name : str
            Name of a prompt class; surrounding whitespace is ignored.

        Returns
        -------
        float
            Score of the class in ``[0, 1]``.

        Raises
        ------
        KeyError
            If the prompt has no class of that name.
        """
        stripped_name: str = class_name.strip()
        if stripped_name not in self.prompt.class_names:
            raise KeyError(f"{stripped_name!r} is not one of {list(self.prompt.class_names)}")
        return float(self.scores[self.prompt.class_names.index(stripped_name)])

    def top_k(self, k: int) -> tuple[Classification, ...]:
        """
        Highest-scoring classes.

        Parameters
        ----------
        k : int
            Number of classes; every class is returned when the prompt has fewer.

        Returns
        -------
        tuple[Classification, ...]
            At most ``k`` classes by descending score; lower class ids first on ties.

        Raises
        ------
        ValueError
            If ``k`` is not positive.
        """
        if k <= 0:
            raise ValueError(f"k must be positive. got {k}")
        return tuple(self.classification(int(class_id)) for class_id in self._ranked_class_ids()[:k])

    def filter_by_score(self, threshold: float) -> tuple[Classification, ...]:
        """
        Classes scoring at least a threshold, e.g. every label present for multi-label SigLIP scores.

        Parameters
        ----------
        threshold : float
            Minimum score in ``[0, 1]``.

        Returns
        -------
        tuple[Classification, ...]
            Classes with ``score >= threshold`` by descending score.

        Raises
        ------
        ValueError
            If ``threshold`` is outside ``[0, 1]``.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]. got {threshold}")
        return tuple(
            self.classification(int(class_id))
            for class_id in self._ranked_class_ids()
            if self.scores[class_id] >= threshold
        )

    def filter_by_logit(self, threshold: float) -> tuple[Classification, ...]:
        """
        Classes whose logit reaches a threshold, independent of the other classes of the prompt.

        A softmax score (CLIP) is relative: one class always takes most of the score, even when the image shows none
        of the classes. The logit combines the scaled cosine similarities of the image and the queries of the class, so
        a threshold on it, chosen on real data, rejects images matching no class and is comparable across prompts of
        the same classifier, templates and combinations. A ``max`` over more candidates is biased upward, so classes
        and prompts with many more queries or sentences per class need a higher threshold.

        Parameters
        ----------
        threshold : float
            Minimum logit.

        Returns
        -------
        tuple[Classification, ...]
            Classes with ``logit >= threshold`` by descending score.

        Raises
        ------
        ValueError
            If ``threshold`` is not finite.
        """
        if not np.isfinite(threshold):
            raise ValueError(f"threshold must be finite. got {threshold}")
        return tuple(
            self.classification(int(class_id))
            for class_id in self._ranked_class_ids()
            if self.class_logits[class_id] >= threshold
        )

    def to_records(self) -> list[ClassificationRecord]:
        """
        Convert every class to plain values, e.g. to write JSON lines or build a data frame.

        Returns
        -------
        list[ClassificationRecord]
            One flat, JSON-serializable record per class, in class-id order.
        """
        return [classification.to_record() for classification in self]

    def _ranked_class_ids(self) -> IntArray:
        return np.argsort(-self.scores, kind="stable").astype(np.int64)
