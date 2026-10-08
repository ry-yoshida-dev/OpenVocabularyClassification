from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

from PIL import Image, ImageOps


class ImageFiles:
    """
    Finds image files in files and directories given on the command line and reads them upright.
    """

    IMAGE_SUFFIXES: ClassVar[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"})

    @classmethod
    def find(cls, paths: Sequence[Path], is_recursive: bool) -> list[Path]:
        """
        Expand files and directories into image files.

        Parameters
        ----------
        paths : Sequence[Path]
            Image files, kept as given, and directories, searched for files with an image suffix.
        is_recursive : bool
            Whether directories are searched with their sub-directories.

        Returns
        -------
        list[Path]
            Image files, each directory's sorted by path.

        Raises
        ------
        FileNotFoundError
            If a path does not exist or no image is found.
        """
        image_paths: list[Path] = []
        for path in paths:
            if path.is_dir():
                candidates: Iterator[Path] = path.rglob("*") if is_recursive else path.iterdir()
                image_paths.extend(
                    sorted(
                        candidate
                        for candidate in candidates
                        if candidate.is_file() and candidate.suffix.lower() in cls.IMAGE_SUFFIXES
                    )
                )
            elif path.is_file():
                image_paths.append(path)
            else:
                raise FileNotFoundError(f"no image file or directory at {path}")
        if not image_paths:
            raise FileNotFoundError(f"no images found in {[str(path) for path in paths]}")
        return image_paths

    @staticmethod
    def load(path: Path) -> Image.Image:
        """
        Read an image into memory upright, closing its file.

        Parameters
        ----------
        path : Path
            Image file.

        Returns
        -------
        Image.Image
            Decoded image rotated as its EXIF orientation says.

        Raises
        ------
        OSError
            If the file cannot be read or decoded as an image.
        """
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image)
