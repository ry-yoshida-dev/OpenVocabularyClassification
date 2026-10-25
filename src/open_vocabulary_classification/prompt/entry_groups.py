from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..array_types import IntArray
from ..options import AggregationUnit
from .embedding_entry import EmbeddingEntry


@dataclass(frozen=True, eq=False)
class EntryGroups:
    """
    The groups one aggregation step forms from the entries of a prompt, numbered by first appearance.

    Entries are ordered by class, so the groups are too, and a step leaving one group per class leaves them in
    class-id order.

    Attributes
    ----------
    group_ids : IntArray
        Group of every entry, shape (E,); read-only.
    merged_entries : tuple[EmbeddingEntry, ...]
        Every group as one entry, in group-id order.
    """

    group_ids: IntArray
    merged_entries: tuple[EmbeddingEntry, ...]

    @classmethod
    def of(cls, entries: Sequence[EmbeddingEntry], unit: AggregationUnit) -> "EntryGroups":
        """
        Group entries by the key of a unit.

        Parameters
        ----------
        entries : Sequence[EmbeddingEntry]
            Entries ordered by class.
        unit : AggregationUnit
            Which entries of a class are merged.

        Returns
        -------
        EntryGroups
            Group of every entry and the merged groups.
        """
        group_of_key: dict[tuple[int, int | None, str | None], int] = {}
        members: list[list[EmbeddingEntry]] = []
        group_ids: list[int] = []
        for entry in entries:
            key: tuple[int, int | None, str | None] = entry.group_key(unit)
            if key not in group_of_key:
                group_of_key[key] = len(members)
                members.append([])
            members[group_of_key[key]].append(entry)
            group_ids.append(group_of_key[key])
        group_array: IntArray = np.array(group_ids, dtype=np.int64)
        group_array.setflags(write=False)
        return cls(group_ids=group_array, merged_entries=tuple(EmbeddingEntry.merged(group, unit) for group in members))
