from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..array_types import FloatArray, IntArray
from ..options import ScoreActivation
from ..prompt import Prompt
from .classification import Classification


@dataclass(frozen=True, eq=False)
class ClassificationResult:
    """
    Model-independent class scores for one image.

    Each class takes the logit of its best query, e.g. ``"puppy"`` for class ``"dog"``, and the activation of the
    classifier turns the class logits into scores. Iterating yields every class in class-id order. The arrays are
    read-only; ``query_logits`` is a copy of the given logits.

    Attributes
    ----------
    prompt : Prompt
        Prompt whose queries were scored.
    query_logits : FloatArray
        Logit of every query, shape (Q,).
    score_activation : ScoreActivation
        Activation turning class logits into scores.
    class_logits : FloatArray
        Logit of the best query of every class, shape (C,).
    matched_query_ids : IntArray
        Best query of every class, indexing ``prompt.queries``, shape (C,).
    scores : FloatArray
        Score of every class in ``[0, 1]``, shape (C,).

    Raises
    ------
    ValueError
        If ``query_logits`` does not hold one finite logit per prompt query.
    """

    prompt: Prompt
    query_logits: FloatArray
    score_activation: ScoreActivation
    class_logits: FloatArray = field(init=False)
    matched_query_ids: IntArray = field(init=False)
    scores: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        expected_shape: tuple[int] = (len(self.prompt.queries),)
        if self.query_logits.shape != expected_shape:
            raise ValueError(f"query_logits must have shape {expected_shape}. got {self.query_logits.shape}")
        if not bool(np.isfinite(self.query_logits).all()):
            raise ValueError("query_logits must be finite.")
        query_logits: FloatArray = self._read_only(self.query_logits.astype(np.float64, copy=True))
        matched_query_ids: IntArray = np.array(
            [query_ids[query_logits[query_ids].argmax()] for query_ids in self.prompt.class_query_ids],
            dtype=np.int64,
        )
        class_logits: FloatArray = query_logits[matched_query_ids]
        object.__setattr__(self, "query_logits", query_logits)
        object.__setattr__(self, "matched_query_ids", self._read_only(matched_query_ids))
        object.__setattr__(self, "class_logits", self._read_only(class_logits))
        object.__setattr__(self, "scores", self._read_only(self.score_activation.apply(class_logits)))

    def __len__(self) -> int:
        return len(self.prompt.class_names)

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
            Score, logit and matched query of the class.

        Raises
        ------
        IndexError
            If ``class_id`` is outside the prompt classes.
        """
        if not 0 <= class_id < len(self):
            raise IndexError(f"class_id must be in [0, {len(self)}). got {class_id}")
        return Classification(
            class_id=class_id,
            class_name=self.prompt.class_names[class_id],
            score=float(self.scores[class_id]),
            logit=float(self.class_logits[class_id]),
            matched_query=self.prompt.queries[int(self.matched_query_ids[class_id])],
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

    def _ranked_class_ids(self) -> IntArray:
        return np.argsort(-self.scores, kind="stable").astype(np.int64)

    @staticmethod
    def _read_only[ArrayT: NDArray[np.generic]](array: ArrayT) -> ArrayT:
        array.setflags(write=False)
        return array
