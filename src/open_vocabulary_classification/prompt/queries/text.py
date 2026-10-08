from dataclasses import dataclass

from ..kind import PromptKind


@dataclass(frozen=True)
class TextQuery:
    """
    Text phrase querying a class, e.g. ``TextQuery("puppy")`` for ``"dog"``.

    Attributes
    ----------
    text : str
        Phrase filled into the text templates; surrounding whitespace is stripped.

    Raises
    ------
    ValueError
        If ``text`` is blank.
    """

    text: str

    def __post_init__(self) -> None:
        stripped_text: str = self.text.strip()
        if not stripped_text:
            raise ValueError("text must not be blank.")
        object.__setattr__(self, "text", stripped_text)

    @property
    def kind(self) -> PromptKind:
        """
        Kind of the query.

        Returns
        -------
        PromptKind
            ``TEXT``.
        """
        return PromptKind.TEXT
