from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import ClassVar, cast

import yaml

from ..options import ClassifierBackend
from .core import ClassifierSettings


class PresetCatalog:
    """
    Reads the YAML presets shipped with the package, and YAML files of the same format, into ``ClassifierSettings``.

    Presets live in ``config/<backend>/<name>.yaml`` inside the package, e.g. ``clip/openai_vit_b16``; each holds the
    settings in its ``classifier`` section.
    """

    PACKAGE: ClassVar[str] = "open_vocabulary_classification"
    PRESET_DIRECTORY: ClassVar[str] = "config"
    PRESET_SUFFIX: ClassVar[str] = ".yaml"
    SECTION: ClassVar[str] = "classifier"

    @classmethod
    def names(cls, backend: ClassifierBackend) -> tuple[str, ...]:
        """
        Names of the presets of one backend.

        Parameters
        ----------
        backend : ClassifierBackend
            Backend whose presets are listed.

        Returns
        -------
        tuple[str, ...]
            Preset names in alphabetical order, e.g. ``("laion_vit_b32", ...)``.
        """
        return tuple(
            sorted(
                entry.name.removesuffix(cls.PRESET_SUFFIX)
                for entry in cls._backend_directory(backend).iterdir()
                if entry.name.endswith(cls.PRESET_SUFFIX)
            )
        )

    @classmethod
    def load(cls, backend: ClassifierBackend, name: str) -> ClassifierSettings:
        """
        Read a preset shipped with the package.

        Parameters
        ----------
        backend : ClassifierBackend
            Backend of the preset.
        name : str
            Preset name, e.g. ``"openai_vit_b16"``.

        Returns
        -------
        ClassifierSettings
            Settings of the preset.

        Raises
        ------
        KeyError
            If the backend has no preset of that name.
        """
        if name not in cls.names(backend):
            raise KeyError(f"{backend} has no preset {name!r}. available: {list(cls.names(backend))}")
        preset: Traversable = cls._backend_directory(backend) / f"{name}{cls.PRESET_SUFFIX}"
        return cls._parse(preset.read_text(encoding="utf-8"), source=f"{backend}/{name}")

    @classmethod
    def load_file(cls, path: Path) -> ClassifierSettings:
        """
        Read a YAML file in the preset format, e.g. a copied and edited preset.

        Parameters
        ----------
        path : Path
            YAML file with a ``classifier`` section.

        Returns
        -------
        ClassifierSettings
            Settings of the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(f"no settings file at {path}")
        return cls._parse(path.read_text(encoding="utf-8"), source=str(path))

    @classmethod
    def _backend_directory(cls, backend: ClassifierBackend) -> Traversable:
        return files(cls.PACKAGE) / cls.PRESET_DIRECTORY / backend.value

    @classmethod
    def _parse(cls, document_text: str, source: str) -> ClassifierSettings:
        document: object = yaml.safe_load(document_text)
        if not isinstance(document, Mapping):
            raise TypeError(f"{source} must hold a mapping. got {type(document).__name__}")
        section: object = cls._string_keyed(cast(Mapping[object, object], document)).get(cls.SECTION)
        if not isinstance(section, Mapping):
            raise KeyError(f"{source} must have a {cls.SECTION!r} mapping.")
        return ClassifierSettings.from_mapping(cls._string_keyed(cast(Mapping[object, object], section)))

    @staticmethod
    def _string_keyed(mapping: Mapping[object, object]) -> dict[str, object]:
        return {str(key): value for key, value in mapping.items()}
