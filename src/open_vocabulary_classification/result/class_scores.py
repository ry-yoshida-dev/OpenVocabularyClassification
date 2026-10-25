from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..array_types import FloatArray, IntArray
from ..options import AggregationMethod, ScoreActivation
from .candidate_logits import CandidateLogits
from .logit_aggregation import LogitAggregation


@dataclass(frozen=True, eq=False)
class ClassScores:
    """
    Score, logit and matched embedding of every class of one image, computed from the logits of its embeddings.

    The aggregation reduces the embedding logits to the candidates of every class, the activation turns the logits
    of every candidate into scores (a softmax shares the score among the candidates of every class), and each class
    takes its best candidate. The arrays are read-only.

    Attributes
    ----------
    class_logits : FloatArray
        Logit of the best candidate of every class, shape (C,).
    matched_entry_ids : IntArray
        Embedding of the highest logit the best candidate of every class was reduced from, shape (C,).
    scores : FloatArray
        Score of every class in ``[0, 1]``, shape (C,).
    """

    class_logits: FloatArray
    matched_entry_ids: IntArray
    scores: FloatArray

    @classmethod
    def of(
        cls, aggregation: LogitAggregation, embedding_logits: FloatArray, score_activation: ScoreActivation
    ) -> "ClassScores":
        """
        Score the classes of one image.

        Parameters
        ----------
        aggregation : LogitAggregation
            How the embedding logits become candidate logits.
        embedding_logits : FloatArray
            Logit of every embedding, shape (E,).
        score_activation : ScoreActivation
            Activation turning candidate logits into scores.

        Returns
        -------
        ClassScores
            Scores in class-id order.

        Raises
        ------
        ValueError
            If ``embedding_logits`` does not have shape (E,).
        """
        candidates: CandidateLogits = aggregation.combine(embedding_logits)
        candidate_scores: FloatArray = score_activation.apply(candidates.logits)
        best_candidate_ids: IntArray = aggregation.class_segments.argmax(candidates.logits)
        return cls(
            class_logits=cls._read_only(candidates.logits[best_candidate_ids]),
            matched_entry_ids=cls._read_only(candidates.best_entry_ids[best_candidate_ids]),
            scores=cls._read_only(aggregation.class_segments.combine(candidate_scores, AggregationMethod.MAX)),
        )

    @staticmethod
    def _read_only[ArrayT: NDArray[np.generic]](array: ArrayT) -> ArrayT:
        array.setflags(write=False)
        return array
