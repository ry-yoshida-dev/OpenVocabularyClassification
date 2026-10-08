from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, eq=False)
class VisualReference:
    """
    Reference image showing an instance of one prompt class, e.g. one photo of a specific mug.

    The whole image is embedded; crop it to the instance when the instance covers only part of it. References compare
    by identity, so a prompt reusing the same reference objects reuses their cached embeddings, which are dropped
    together with the reference.

    Attributes
    ----------
    image : Image.Image
        Reference image of any size and mode.
    """

    image: Image.Image
