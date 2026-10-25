from collections.abc import Sequence
from dataclasses import dataclass

from ..options import AggregationUnit


@dataclass(frozen=True)
class EmbeddingEntry:
    """
    One embedding of a class, or a group of them merged by aggregation steps, labeled by what its members share.

    Before any aggregation an entry is one cell of the class table: a text query filled into one template, or a
    visual query. Merging along the template axis drops the template label, merging along the prompt axis drops the
    query label, however many members a group has, so later steps group by the remaining axis.

    Attributes
    ----------
    class_id : int
        Class of every member.
    query_id : int | None
        Query every member belongs to; ``None`` once the prompt axis is merged.
    template : str | None
        Template of every member; ``None`` once the template axis is merged, and for a visual query.
    sentences : tuple[str, ...]
        Text members: a text query filled into a template each.
    visual_query_ids : tuple[int, ...]
        Visual members: queries whose references are averaged into one embedding each.

    Raises
    ------
    ValueError
        If ``class_id`` is negative or the entry has no member.
    """

    class_id: int
    query_id: int | None
    template: str | None
    sentences: tuple[str, ...]
    visual_query_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError(f"class_id must not be negative. got {self.class_id}")
        if not self.sentences and not self.visual_query_ids:
            raise ValueError("an entry needs at least one sentence or visual query.")

    def group_key(self, unit: AggregationUnit) -> tuple[int, int | None, str | None]:
        """
        Key shared by the entries one aggregation step merges with this one.

        Parameters
        ----------
        unit : AggregationUnit
            Which entries of a class are merged.

        Returns
        -------
        tuple[int, int | None, str | None]
            Class with the query for ``TEMPLATE``, with the template for ``PROMPT``, alone otherwise.
        """
        match unit:
            case AggregationUnit.TEMPLATE:
                return (self.class_id, self.query_id, None)
            case AggregationUnit.PROMPT:
                return (self.class_id, None, self.template)
            case AggregationUnit.TEMPLATE_AND_PROMPT:
                return (self.class_id, None, None)

    @classmethod
    def merged(cls, entries: Sequence["EmbeddingEntry"], unit: AggregationUnit) -> "EmbeddingEntry":
        """
        Merge the entries of one group of an aggregation step into one entry holding all their members.

        Parameters
        ----------
        entries : Sequence[EmbeddingEntry]
            Entries sharing ``group_key(unit)``.
        unit : AggregationUnit
            Which axis the step merges.

        Returns
        -------
        EmbeddingEntry
            Entry keeping the group key as its labels.

        Raises
        ------
        ValueError
            If ``entries`` is empty or the entries do not share the group key.
        """
        if not entries or len({entry.group_key(unit) for entry in entries}) != 1:
            raise ValueError("entries to merge must be non-empty and share their group key.")
        class_id, query_id, template = entries[0].group_key(unit)
        return cls(
            class_id=class_id,
            query_id=query_id,
            template=template,
            sentences=tuple(sentence for entry in entries for sentence in entry.sentences),
            visual_query_ids=tuple(query_id for entry in entries for query_id in entry.visual_query_ids),
        )
