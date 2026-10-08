from enum import StrEnum


class OutputFormat(StrEnum):
    """
    How the command line prints the classes of each image.

    Attributes
    ----------
    TEXT : str
        Human-readable lines: the image name, then one indented line per class.
    JSON_LINES : str
        One JSON object per image with its path and the ``ClassificationRecord`` of each class, for other programs.
    """

    TEXT = "text"
    JSON_LINES = "jsonl"
