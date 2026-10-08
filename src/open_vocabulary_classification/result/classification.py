from dataclasses import dataclass

from ..prompt import PromptQuery


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
        Logit of the best query of the class, comparable across prompts of the same classifier.
    matched_query : PromptQuery
        Query of the class that scored highest, e.g. ``TextQuery("puppy")`` for class ``"dog"``.
    """

    class_id: int
    class_name: str
    score: float
    logit: float
    matched_query: PromptQuery
