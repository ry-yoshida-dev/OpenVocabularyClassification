from enum import StrEnum


class PromptKind(StrEnum):
    """
    Kind of prompt query, i.e. which input tells the model what a class looks like.

    A prompt may mix both kinds, even within one class.

    Attributes
    ----------
    TEXT : str
        Query by a text phrase.
    VISUAL : str
        Query by reference images.
    """

    TEXT = "text"
    VISUAL = "visual"
