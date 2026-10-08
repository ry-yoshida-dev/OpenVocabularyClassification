from dataclasses import dataclass

from ..kind import PromptKind
from ..visual_reference import VisualReference


@dataclass(frozen=True)
class VisualQuery:
    """
    Reference images querying a class together, e.g. several photos of the same mug (few-shot prototype).

    The L2-normalized image embeddings of the references are averaged into one query. Give appearances that differ
    (a sedan and a van) as separate queries instead, so each is scored on its own.

    Attributes
    ----------
    references : tuple[VisualReference, ...]
        Reference images of the queried class.

    Raises
    ------
    ValueError
        If there is no reference or one reference is given twice.
    """

    references: tuple[VisualReference, ...]

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("a visual query needs at least one reference.")
        if len(set(self.references)) != len(self.references):
            raise ValueError("a visual query must not repeat a reference.")

    @property
    def kind(self) -> PromptKind:
        """
        Kind of the query.

        Returns
        -------
        PromptKind
            ``VISUAL``.
        """
        return PromptKind.VISUAL
