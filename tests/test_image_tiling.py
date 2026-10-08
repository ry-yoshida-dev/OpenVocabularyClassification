import pytest
from PIL import Image

from open_vocabulary_classification import ImageTiling


def test_tiles_cover_the_image_with_overlap() -> None:
    image: Image.Image = Image.new("RGB", (100, 60))
    tiling: ImageTiling = ImageTiling(grid_size=2, overlap_ratio=0.25)
    views: list[Image.Image] = tiling.views(image)
    assert len(views) == tiling.view_count == 5
    assert views[0] is image
    tile_width: float = 100 / 1.75
    assert [view.size for view in views[1:]] == [(round(tile_width), round(60 / 1.75))] * 4


def test_tiles_without_full_image_and_overlap_partition_the_image() -> None:
    image: Image.Image = Image.new("RGB", (90, 30))
    for x in range(90):
        image.putpixel((x, 0), (x, 0, 0))
    tiling: ImageTiling = ImageTiling(grid_size=3, overlap_ratio=0.0, is_full_image_included=False)
    views: list[Image.Image] = tiling.views(image)
    assert tiling.view_count == len(views) == 9
    assert [view.size for view in views] == [(30, 10)] * 9
    assert [views[column].getpixel((0, 0)) for column in range(3)] == [(0, 0, 0), (30, 0, 0), (60, 0, 0)]


def test_tiles_of_tiny_images_are_never_empty() -> None:
    views: list[Image.Image] = ImageTiling(grid_size=4).views(Image.new("RGB", (2, 1)))
    assert all(view.width >= 1 and view.height >= 1 for view in views)


@pytest.mark.parametrize(("grid_size", "overlap_ratio"), [(1, 0.0), (2, -0.1), (2, 1.0)])
def test_invalid_tiling_raises(grid_size: int, overlap_ratio: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        ImageTiling(grid_size=grid_size, overlap_ratio=overlap_ratio)
