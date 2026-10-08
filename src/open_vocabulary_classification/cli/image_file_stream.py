from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from PIL import Image

from .image_files import ImageFiles


@dataclass
class ImageFileStream:
    """
    Reads image files one at a time while they are consumed, skipping files that cannot be decoded.

    A skipped file is reported on ``warnings`` and recorded in ``skipped_paths``, so one corrupt file does not stop a
    run over a whole directory. The path of every yielded image is queued in ``pending_paths``; a consumer producing
    one result per image in input order pops the path of each result from its front.

    Attributes
    ----------
    paths : Sequence[Path]
        Image files to read, in order.
    warnings : TextIO
        Stream skipped files are reported on.
    pending_paths : deque[Path]
        Paths of the images yielded and not yet popped by the consumer, oldest first.
    skipped_paths : list[Path]
        Files that could not be read, in order.
    """

    paths: Sequence[Path]
    warnings: TextIO
    pending_paths: deque[Path] = field(default_factory=deque[Path], init=False)
    skipped_paths: list[Path] = field(default_factory=list[Path], init=False)

    def __iter__(self) -> Iterator[Image.Image]:
        for path in self.paths:
            try:
                image: Image.Image = ImageFiles.load(path)
            except (OSError, Image.DecompressionBombError) as error:
                self.skipped_paths.append(path)
                self.warnings.write(f"skipping {path}: {error}\n")
                continue
            self.pending_paths.append(path)
            yield image
