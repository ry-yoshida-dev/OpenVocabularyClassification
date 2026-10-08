from collections.abc import Mapping, Sequence

import pytest
from PIL import Image

from open_vocabulary_classification import Prompt, PromptKind, PromptQuery, TextQuery, VisualQuery, VisualReference


def build_reference() -> VisualReference:
    return VisualReference(image=Image.new("RGB", (100, 50)))


def test_text_prompt_classes_follow_the_keys() -> None:
    prompt: Prompt = Prompt.from_texts({" dog ": ("dog", " puppy", "hound "), "cat": ("cat",)})
    assert prompt.kinds == frozenset({PromptKind.TEXT})
    assert prompt.class_names == ("dog", "cat")
    assert prompt.queries == (TextQuery("dog"), TextQuery("puppy"), TextQuery("hound"), TextQuery("cat"))
    assert prompt.query_class_ids == (0, 0, 0, 1)
    assert [query_ids.tolist() for query_ids in prompt.class_query_ids] == [[0, 1, 2], [3]]


def test_class_names_query_themselves() -> None:
    prompt: Prompt = Prompt.from_class_names((" cat", "traffic light"))
    assert prompt == Prompt.from_texts({"cat": ("cat",), "traffic light": ("traffic light",)})
    assert prompt.query_class_ids == (0, 1)


def test_one_class_mixes_text_and_visual_queries() -> None:
    my_dog_references: tuple[VisualReference, ...] = (build_reference(), build_reference())
    prompt: Prompt = Prompt(
        {
            "cat": (TextQuery("cat"),),
            "dog": (TextQuery("dog"), VisualQuery(my_dog_references)),
        }
    )
    assert prompt.kinds == frozenset({PromptKind.TEXT, PromptKind.VISUAL})
    assert prompt.query_class_ids == (0, 1, 1)
    assert [query_ids.tolist() for query_ids in prompt.class_query_ids] == [[0], [1, 2]]
    assert prompt.queries[2] == VisualQuery(my_dog_references)


def test_prompt_copies_its_input_and_compares_by_value() -> None:
    class_texts: dict[str, tuple[str, ...]] = {"dog": ("puppy",)}
    prompt: Prompt = Prompt.from_texts(class_texts)
    class_texts["dog"] = ("hound",)
    assert prompt.queries == (TextQuery("puppy"),)
    assert prompt == Prompt.from_texts({"dog": ("puppy",)})
    assert hash(prompt) == hash(Prompt.from_texts({"dog": ("puppy",)}))
    assert prompt != Prompt.from_texts({"canine": ("puppy",)})


def test_visual_queries_compare_by_reference_identity() -> None:
    reference: VisualReference = build_reference()
    same_pixels: VisualReference = build_reference()
    assert Prompt({"mug": (VisualQuery((reference,)),)}) == Prompt({"mug": (VisualQuery((reference,)),)})
    assert Prompt({"mug": (VisualQuery((reference,)),)}) != Prompt({"mug": (VisualQuery((same_pixels,)),)})


REFERENCE: VisualReference = build_reference()
OTHER_REFERENCE: VisualReference = build_reference()


@pytest.mark.parametrize(
    ("class_queries", "message"),
    [
        ({}, "at least one class"),
        ({" ": (TextQuery("x"),)}, "must not be blank"),
        ({"dog": (TextQuery("dog"),), "dog ": (TextQuery("puppy"),)}, "class names must be unique"),
        ({"dog": ()}, "needs at least one query"),
        ({"dog": (TextQuery("puppy"), TextQuery("Puppy"))}, "more than once"),
        ({"dog": (TextQuery("pet"),), "cat": (TextQuery("pet"),)}, "more than once"),
        ({"mug": (VisualQuery((REFERENCE,)), VisualQuery((REFERENCE,)))}, "more than once"),
        ({"mug": (VisualQuery((REFERENCE,)),), "cup": (VisualQuery((REFERENCE, OTHER_REFERENCE)),)}, "only one class"),
    ],
)
def test_invalid_prompts_raise(class_queries: Mapping[str, Sequence[PromptQuery]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Prompt(class_queries)


def test_query_validation() -> None:
    with pytest.raises(ValueError, match="blank"):
        TextQuery(" ")
    assert TextQuery(" puppy ").text == "puppy"
    with pytest.raises(ValueError, match="at least one reference"):
        VisualQuery(())
    with pytest.raises(ValueError, match="must not repeat"):
        VisualQuery((REFERENCE, REFERENCE))
