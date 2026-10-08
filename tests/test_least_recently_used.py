import pytest

from open_vocabulary_classification.cache import LeastRecentlyUsedMapping


def test_least_recently_used_entry_is_evicted() -> None:
    mapping: LeastRecentlyUsedMapping[str, int] = LeastRecentlyUsedMapping(2)
    mapping["a"] = 1
    mapping["b"] = 2
    assert mapping["a"] == 1
    mapping["c"] = 3
    assert list(mapping) == ["a", "c"]
    assert len(mapping) == mapping.capacity == 2


def test_overwriting_refreshes_an_entry() -> None:
    mapping: LeastRecentlyUsedMapping[str, int] = LeastRecentlyUsedMapping(2)
    mapping.update({"a": 1, "b": 2})
    mapping["a"] = 10
    mapping["c"] = 3
    assert dict(mapping) == {"a": 10, "c": 3}
    del mapping["a"]
    assert "a" not in mapping


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        LeastRecentlyUsedMapping[str, int](0)
