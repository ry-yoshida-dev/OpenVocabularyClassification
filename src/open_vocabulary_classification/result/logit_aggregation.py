from dataclasses import dataclass, field

import numpy as np

from ..array_types import FloatArray, IntArray
from ..options import LogitCombination, TemplateCombination
from ..prompt import EmbeddingLayout
from .class_logits import ClassLogits
from .logit_segments import LogitSegments


@dataclass(frozen=True, eq=False)
class LogitAggregation:
    """
    How the embedding logits of a prompt layout become class logits, prepared once per prompt.

    ``template_combination`` turns the sentence logits of each text query into one query logit with ``MAX``, ``MEAN``,
    ``MEDIAN`` or ``MIN``; ``SEPARATE`` keeps every sentence as a candidate of its own and ``ENSEMBLE`` has a single
    embedding per query already. ``query_combination`` then turns the candidates of a class into the class logit. A
    layout groups its embeddings by query and a prompt its queries by class, so both steps are reductions over
    consecutive logits and scoring an image costs a few vectorized NumPy calls, whatever the number of classes,
    queries and templates.

    Attributes
    ----------
    layout : EmbeddingLayout
        Embeddings of the prompt that are scored.
    template_combination : TemplateCombination
        How the sentences of a text query are combined; must match how ``layout`` was laid out.
    query_combination : LogitCombination
        How the candidates of a class become its logit.
    query_segments : LogitSegments
        Embeddings of every query.
    class_segments : LogitSegments
        Candidates of every class: its queries, or the sentences of its queries for ``SEPARATE``.

    Raises
    ------
    ValueError
        If ``layout`` ensembles its text queries but ``template_combination`` is not ``ENSEMBLE``, or the other way
        round.
    """

    layout: EmbeddingLayout
    template_combination: TemplateCombination = TemplateCombination.ENSEMBLE
    query_combination: LogitCombination = LogitCombination.MAX
    query_segments: LogitSegments = field(init=False)
    class_segments: LogitSegments = field(init=False)

    def __post_init__(self) -> None:
        is_ensembled: bool = self.template_combination.is_ensembled
        if any(bool(entry.sentences) and (entry.template is None) != is_ensembled for entry in self.layout.entries):
            raise ValueError(f"layout does not match template_combination {self.template_combination}.")
        query_class_ids: IntArray = np.array(self.layout.prompt.query_class_ids, dtype=np.int64)
        candidate_class_ids: IntArray
        match self.template_combination.logit_combination:
            case None:
                candidate_class_ids = query_class_ids[self.layout.entry_query_ids]
            case LogitCombination():
                candidate_class_ids = query_class_ids
        object.__setattr__(self, "query_segments", LogitSegments.of(self.layout.entry_query_ids))
        object.__setattr__(self, "class_segments", LogitSegments.of(candidate_class_ids))

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
            Logit of every class and the embedding it matched best: the highest sentence of its highest candidate.

        Raises
        ------
        ValueError
            If ``embedding_logits`` does not have shape (E,).
        """
        candidate_logits: FloatArray
        candidate_entry_ids: IntArray
        sentence_combination: LogitCombination | None = self.template_combination.logit_combination
        match sentence_combination:
            case None:
                candidate_logits = embedding_logits
                candidate_entry_ids = np.arange(embedding_logits.size, dtype=np.int64)
            case LogitCombination():
                candidate_logits = self.query_segments.combine(embedding_logits, sentence_combination)
                candidate_entry_ids = self.query_segments.argmax(embedding_logits)
        return ClassLogits(
            logits=self.class_segments.combine(candidate_logits, self.query_combination),
            matched_entry_ids=candidate_entry_ids[self.class_segments.argmax(candidate_logits)],
        )
