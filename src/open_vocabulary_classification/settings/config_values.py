from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast


@dataclass(frozen=True)
class ConfigValues:
    """
    Typed, validating access to the plain values of a configuration section, e.g. parsed YAML.

    Every accessor raises ``KeyError`` for a missing required key and ``TypeError`` for a value of the wrong type, so
    a malformed configuration fails before a model is loaded.

    Attributes
    ----------
    values : Mapping[str, object]
        Values keyed by field name.
    """

    values: Mapping[str, object]

    def validate_keys(self, known_keys: frozenset[str]) -> None:
        """
        Reject keys that no field reads, e.g. a misspelled field name.

        Parameters
        ----------
        known_keys : frozenset[str]
            Keys that may be given.

        Raises
        ------
        KeyError
            If a key is not one of ``known_keys``.
        """
        unknown_keys: list[str] = sorted(set(self.values) - known_keys)
        if unknown_keys:
            raise KeyError(f"unknown configuration keys {unknown_keys}. known: {sorted(known_keys)}")

    def text(self, key: str) -> str:
        """
        Read a required string.

        Parameters
        ----------
        key : str
            Field name.

        Returns
        -------
        str
            Value of the field.

        Raises
        ------
        KeyError
            If the key is missing.
        TypeError
            If the value is not a string.
        """
        value: object = self._required(key)
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string. got {type(value).__name__}")
        return value

    def texts(self, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        """
        Read an optional list of strings.

        Parameters
        ----------
        key : str
            Field name.
        default : tuple[str, ...]
            Value when the key is missing.

        Returns
        -------
        tuple[str, ...]
            Strings in the given order.

        Raises
        ------
        TypeError
            If the value is not a list or tuple of strings.
        """
        if key not in self.values:
            return default
        value: object = self.values[key]
        if not isinstance(value, list | tuple):
            raise TypeError(f"{key} must be a list of strings. got {type(value).__name__}")
        items: list[object] = list(cast(Sequence[object], value))
        strings: list[str] = [item for item in items if isinstance(item, str)]
        if len(strings) != len(items):
            raise TypeError(f"{key} must be a list of strings. got {items}")
        return tuple(strings)

    def integer(self, key: str, default: int) -> int:
        """
        Read an optional integer; booleans are rejected.

        Parameters
        ----------
        key : str
            Field name.
        default : int
            Value when the key is missing.

        Returns
        -------
        int
            Value of the field.

        Raises
        ------
        TypeError
            If the value is not an integer.
        """
        if key not in self.values:
            return default
        value: object = self.values[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer. got {type(value).__name__}")
        return value

    def member[EnumT: Enum](self, key: str, enum_type: type[EnumT], default: EnumT | None = None) -> EnumT:
        """
        Read an enum member given by its value.

        Parameters
        ----------
        key : str
            Field name.
        enum_type : type[EnumT]
            Enum the value belongs to.
        default : EnumT | None
            Value when the key is missing; ``None`` makes the key required.

        Returns
        -------
        EnumT
            Member whose value is the given value.

        Raises
        ------
        KeyError
            If a required key is missing.
        ValueError
            If the value is not the value of a member.
        """
        if default is not None and key not in self.values:
            return default
        value: object = self._required(key)
        for member in enum_type:
            if member.value == value:
                return member
        raise ValueError(f"{key} must be one of {[member.value for member in enum_type]}. got {value!r}")

    def _required(self, key: str) -> object:
        if key not in self.values:
            raise KeyError(f"missing required configuration key {key!r}")
        return self.values[key]
