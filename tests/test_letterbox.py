import pytest
from PIL import Image

from open_vocabulary_classification import Letterbox


@pytest.mark.parametrize(("size", "offset"), [((6, 2), (0, 2)), ((2, 6), (2, 0))])
def test_non_square_images_are_centered_on_a_square(size: tuple[int, int], offset: tuple[int, int]) -> None:
    image: Image.Image = Image.new("RGB", size, (200, 100, 50))
    square: Image.Image = Letterbox((1, 2, 3)).apply(image)
    assert square.size == (6, 6)
    assert square.getpixel(offset) == (200, 100, 50)
    assert square.getpixel((0, 0)) == (1, 2, 3)
    assert square.getpixel((5, 5)) == (1, 2, 3)


def test_square_images_are_returned_unchanged() -> None:
    image: Image.Image = Image.new("RGB", (4, 4))
    assert Letterbox((0, 0, 0)).apply(image) is image


def test_fill_color_outside_the_channel_range_raises() -> None:
    with pytest.raises(ValueError, match="fill_color"):
        Letterbox((0, 256, 0))
