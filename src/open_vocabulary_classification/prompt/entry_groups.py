from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..array_types import IntArray
from ..options import AggregationUnit
from .entry_label import EntryLabel


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
    labels : tuple[EntryLabel, ...]
        Merged label of every group, in group-id order.
    """

    group_ids: IntArray
    labels: tuple[EntryLabel, ...]

    @classmethod
    def of(cls, labels: Sequence[EntryLabel], unit: AggregationUnit) -> "EntryGroups":
        """
        Group entries by their label merged by a unit.

        Parameters
        ----------
        labels : Sequence[EntryLabel]
            Labels of the entries, ordered by class.
        unit : AggregationUnit
            Which entries of a class are merged.

        Returns
        -------
        EntryGroups
            Group of every entry and the label of every group.
        """
        group_of_label: dict[EntryLabel, int] = {}
        group_ids: list[int] = []
        for label in labels:
            group_ids.append(group_of_label.setdefault(label.merged(unit), len(group_of_label)))
        group_array: IntArray = np.array(group_ids, dtype=np.int64)
        group_array.setflags(write=False)
        return cls(group_ids=group_array, labels=tuple(group_of_label))

    def members_of[ItemT](self, items: Sequence[ItemT]) -> tuple[tuple[ItemT, ...], ...]:
        """
        Split items of the entries by group.

        Parameters
        ----------
        items : Sequence[ItemT]
            One item per entry, in entry order, e.g. the entries themselves.

        Returns
        -------
        tuple[tuple[ItemT, ...], ...]
            Items of every group in group-id order, entry order within a group.

        Raises
        ------
        ValueError
            If ``items`` does not hold one item per entry.
        """
        if len(items) != self.group_ids.size:
            raise ValueError(f"items must hold one item per entry ({self.group_ids.size}). got {len(items)}")
        members: list[list[ItemT]] = [[] for _ in self.labels]
        for item, group_id in zip(items, self.group_ids.tolist(), strict=True):
            members[group_id].append(item)
        return tuple(tuple(group) for group in members)
