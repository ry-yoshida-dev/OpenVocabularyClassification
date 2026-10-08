from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ImageTiling:
    """
    Splits an image into a grid of overlapping tiles, each embedded at the full input resolution of the model.

    Models resize every image to a small input (e.g. 224 pixels), so small objects in large images lose most of their
    pixels. Tiling classifies each tile, and the full image when ``is_full_image_included``, and keeps the highest
    logit of every query over these views, so an object visible in any one view is found.

    Attributes
    ----------
    grid_size : int
        Number of tiles along each side; ``2`` gives four tiles.
    overlap_ratio : float
        Share of a tile overlapping its neighbor, in ``[0, 1)``, so objects on a tile border appear whole in one tile.
    is_full_image_included : bool
        Whether the whole image is a view too, so objects larger than a tile are still seen whole.

    Raises
    ------
    ValueError
        If ``grid_size`` is below 2 or ``overlap_ratio`` is outside ``[0, 1)``.
    """

    grid_size: int = 2
    overlap_ratio: float = 0.25
    is_full_image_included: bool = True

    def __post_init__(self) -> None:
        if self.grid_size < 2:
            raise ValueError(f"grid_size must be at least 2. got {self.grid_size}")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0, 1). got {self.overlap_ratio}")

    @property
    def view_count(self) -> int:
        """
        Number of views per image.

        Returns
        -------
        int
            ``grid_size ** 2`` tiles, plus one for the full image when it is included.
        """
        return self.grid_size**2 + int(self.is_full_image_included)

    def views(self, image: Image.Image) -> list[Image.Image]:
        """
        Cut an image into its views.

        Parameters
        ----------
        image : Image.Image
            Image to cut.

        Returns
        -------
        list[Image.Image]
            The full image first when included, then the tiles row by row; ``view_count`` views.
        """
        column_bounds: list[tuple[int, int]] = self._tile_bounds(image.width)
        row_bounds: list[tuple[int, int]] = self._tile_bounds(image.height)
        tiles: list[Image.Image] = [
            image.crop((left, top, right, bottom)) for top, bottom in row_bounds for left, right in column_bounds
        ]
        return [image, *tiles] if self.is_full_image_included else tiles

    def _tile_bounds(self, length: int) -> list[tuple[int, int]]:
        tile_length: float = length / (self.grid_size - (self.grid_size - 1) * self.overlap_ratio)
        step: float = tile_length * (1.0 - self.overlap_ratio)
        bounds: list[tuple[int, int]] = []
        for index in range(self.grid_size):
            start: int = min(round(index * step), length - 1)
            end: int = max(min(round(index * step + tile_length), length), start + 1)
            bounds.append((start, end))
        return bounds
