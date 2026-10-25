from typing import TypedDict


class ClassificationRecord(TypedDict):
    """
    JSON-serializable form of a ``Classification``, flat so it also fits a CSV row or a data frame.

    Attributes
    ----------
    class_id : int
        Index into the classes of the originating prompt.
    class_name : str
        Class name corresponding to ``class_id``.
    score : float
        Score in ``[0, 1]``.
    logit : float
        Logit aggregated from the embeddings of the class.
    matched_query_kind : str | None
        ``PromptKind`` value of the matched query, ``"text"`` or ``"visual"``; ``None`` without a single matched query.
    matched_query_text : str | None
        Phrase of a matched text query; ``None`` otherwise.
    matched_query_reference_count : int | None
        Number of reference images of a matched visual query; ``None`` otherwise.
    matched_template : str | None
        Template of the matched embedding; ``None`` without a single template.
    """

    class_id: int
    class_name: str
    score: float
    logit: float
    matched_query_kind: str | None
    matched_query_text: str | None
    matched_query_reference_count: int | None
    matched_template: str | None
