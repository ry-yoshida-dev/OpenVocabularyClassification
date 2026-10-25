from dataclasses import dataclass

from ..options import AggregationUnit


@dataclass(frozen=True)
class EntryLabel:
    """
    What the members of one embedding, or of a group merged by aggregation steps, share.

    Labels are all an aggregation step groups by, so logits stored with the labels of their embeddings can be
    aggregated again without the prompt, e.g. with other ``SIMILARITY`` steps.

    Attributes
    ----------
    class_id : int
        Class of every member.
    query_id : int | None
        Query every member belongs to; ``None`` once the prompt axis is merged.
    template : str | None
        Template of every member; ``None`` once the template axis is merged, and for a visual query.

    Raises
    ------
    ValueError
        If ``class_id`` is negative.
    """

    class_id: int
    query_id: int | None
    template: str | None

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError(f"class_id must not be negative. got {self.class_id}")

    def merged(self, unit: AggregationUnit) -> "EntryLabel":
        """
        Label of the group one aggregation step merges this entry into.

        Parameters
        ----------
        unit : AggregationUnit
            Which entries of a class are merged.

        Returns
        -------
        EntryLabel
            Label keeping the class with the query for ``TEMPLATE``, with the template for ``PROMPT``, alone otherwise.
        """
        match unit:
            case AggregationUnit.TEMPLATE:
                return EntryLabel(self.class_id, self.query_id, None)
            case AggregationUnit.PROMPT:
                return EntryLabel(self.class_id, None, self.template)
            case AggregationUnit.TEMPLATE_AND_PROMPT:
                return EntryLabel(self.class_id, None, None)
