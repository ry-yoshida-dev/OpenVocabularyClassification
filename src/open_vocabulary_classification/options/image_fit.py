from enum import StrEnum


class ImageFit(StrEnum):
    """
    How an image of any aspect ratio is fitted into the input of the model.

    Attributes
    ----------
    LETTERBOX : str
        Pad the image to a square with the mean color of the model before its processor resizes it, so the whole image
        is seen without distortion. Models keeping the aspect ratio by themselves (SigLIP 2 NaFlex) are fed the image
        as it is.
    NATIVE : str
        The preprocessing of the checkpoint: CLIP resizes the shorter side and crops the center, cutting off the edges
        of non-square images; fixed-resolution SigLIP stretches the image to a square; SigLIP 2 NaFlex keeps the
        aspect ratio.
    """

    LETTERBOX = "letterbox"
    NATIVE = "native"
