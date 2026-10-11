from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Letterbox:
    """
    Pads images to a square, centered, so a square model input shows the whole image without distortion.

    Only padding is added; the processor of the model resizes the square afterwards, so the image is resampled once.

    Attributes
    ----------
    fill_color : tuple[int, int, int]
        RGB color of the padding, e.g. the mean color the model normalizes by, so the padding is zero after
        normalization.

    Raises
    ------
    ValueError
        If a channel of ``fill_color`` is outside ``[0, 255]``.
    """

    fill_color: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not all(0 <= channel <= 255 for channel in self.fill_color):
            raise ValueError(f"fill_color channels must be in [0, 255]. got {self.fill_color}")

    def apply(self, image: Image.Image) -> Image.Image:
        """
        Pad an RGB image to a square.

        Parameters
        ----------
        image : Image.Image
            RGB image.

        Returns
        -------
        Image.Image
            The image itself when it is square, otherwise a square of side ``max(width, height)`` with the image
            centered on ``fill_color``.
        """
        if image.width == image.height:
            return image
        side: int = max(image.width, image.height)
        square: Image.Image = Image.new("RGB", (side, side), self.fill_color)
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return square
