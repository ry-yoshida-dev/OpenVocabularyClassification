import pytest

from open_vocabulary_classification import TextTemplates


def test_text_query_fills_every_template_in_order() -> None:
    templates: TextTemplates = TextTemplates(("a photo of a {}.", "{}", "a {{sketch}} of a {}."))
    assert len(templates) == 3
    assert templates.fill("dog") == ("a photo of a dog.", "dog", "a {sketch} of a dog.")


def test_braces_in_the_text_query_are_kept() -> None:
    assert TextTemplates(("a photo of {}.",)).fill("{x}") == ("a photo of {x}.",)


@pytest.mark.parametrize(
    ("templates", "message"),
    [
        ((), "at least one text template"),
        (("a {}", "a {}"), "must be unique"),
        (("a photo",), "exactly one bare"),
        (("{} and {}",), "exactly one bare"),
        (("a {name}",), "exactly one bare"),
        (("a {0}",), "exactly one bare"),
        (("a {:>5}",), "exactly one bare"),
        (("a {!r}",), "exactly one bare"),
        (("a {",), "Single '{'"),
    ],
)
def test_invalid_templates_raise(templates: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TextTemplates(templates)
