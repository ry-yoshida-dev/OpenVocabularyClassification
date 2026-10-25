from dataclasses import dataclass

from ..prompt import PromptQuery, TextQuery, VisualQuery
from .classification_record import ClassificationRecord


@dataclass(frozen=True)
class Classification:
    """
    Score of one prompt class for one image.

    Attributes
    ----------
    class_id : int
        Index into the classes of the originating prompt.
    class_name : str
        Class name corresponding to ``class_id``.
    score : float
        Score in ``[0, 1]``: a share of the prompt classes for CLIP, an independent probability for SigLIP.
    logit : float
        Logit combined from the queries of the class, comparable across prompts of the same classifier.
    matched_query : PromptQuery
        Query of the class that scored highest, e.g. ``TextQuery("puppy")`` for class ``"dog"``.
    matched_template : str | None
        Template of the highest sentence of ``matched_query``, e.g. ``"a photo of a {}."``; ``None`` when the
        sentences were ensembled into one embedding or the query is visual.
    """

    class_id: int
    class_name: str
    score: float
    logit: float
    matched_query: PromptQuery
    matched_template: str | None

    def to_record(self) -> ClassificationRecord:
        """
        Convert to plain values, e.g. to write JSON lines.

        Returns
        -------
        ClassificationRecord
            Flat, JSON-serializable copy; reference images are summarized by their count.
        """
        match self.matched_query:
            case TextQuery(text=text):
                matched_query_text: str | None = text
                matched_query_reference_count: int | None = None
            case VisualQuery(references=references):
                matched_query_text = None
                matched_query_reference_count = len(references)
        return ClassificationRecord(
            class_id=self.class_id,
            class_name=self.class_name,
            score=self.score,
            logit=self.logit,
            matched_query_kind=self.matched_query.kind.value,
            matched_query_text=matched_query_text,
            matched_query_reference_count=matched_query_reference_count,
            matched_template=self.matched_template,
        )
