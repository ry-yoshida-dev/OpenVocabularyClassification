from collections import OrderedDict
from collections.abc import Hashable, Iterator, MutableMapping


class LeastRecentlyUsedMapping[KeyT: Hashable, ValueT](MutableMapping[KeyT, ValueT]):
    """
    Mapping holding at most ``capacity`` entries, evicting the entry read or written least recently.

    Used as ``EmbeddingCache`` storage when keys are unbounded, e.g. text queries of a long-running service, so memory
    stays bounded while frequently used keys stay cached.
    """

    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of entries.

        Raises
        ------
        ValueError
            If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive. got {capacity}")
        self._capacity: int = capacity
        self._entries: OrderedDict[KeyT, ValueT] = OrderedDict()

    @property
    def capacity(self) -> int:
        """
        Maximum number of entries.

        Returns
        -------
        int
            Capacity given at construction.
        """
        return self._capacity

    def __getitem__(self, key: KeyT) -> ValueT:
        value: ValueT = self._entries[key]
        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def __delitem__(self, key: KeyT) -> None:
        del self._entries[key]

    def __iter__(self) -> Iterator[KeyT]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
