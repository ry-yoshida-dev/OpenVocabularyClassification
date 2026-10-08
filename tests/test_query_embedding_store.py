import gc
from collections.abc import Sequence

import torch
from PIL import Image

from open_vocabulary_classification import Prompt, TextQuery, TextTemplates, VisualQuery, VisualReference
from open_vocabulary_classification.cache import QueryEmbeddingStore

SENTENCE_EMBEDDINGS: dict[str, list[float]] = {
    "cat": [3.0, 0.0],
    "dog": [0.0, 2.0],
    "bird": [1.0, 1.0],
    "a cat": [0.0, 5.0],
    "a dog": [4.0, 0.0],
}


class CountingEmbedder:
    def __init__(self) -> None:
        self.embedded_sentences: list[list[str]] = []
        self.embedded_references: list[VisualReference] = []

    def embed_sentences(self, sentences: Sequence[str]) -> torch.Tensor:
        self.embedded_sentences.append(list(sentences))
        return torch.tensor([SENTENCE_EMBEDDINGS[sentence] for sentence in sentences])

    def embed_references(self, references: Sequence[VisualReference]) -> torch.Tensor:
        self.embedded_references.extend(references)
        return torch.tensor([[float(reference.image.width), float(reference.image.height)] for reference in references])


def build_store(
    templates: tuple[str, ...] = ("{}",), text_cache_capacity: int = 16
) -> tuple[QueryEmbeddingStore, CountingEmbedder]:
    embedder: CountingEmbedder = CountingEmbedder()
    store: QueryEmbeddingStore = QueryEmbeddingStore(
        embed_sentences=embedder.embed_sentences,
        embed_references=embedder.embed_references,
        text_templates=TextTemplates(templates),
        text_cache_capacity=text_cache_capacity,
    )
    return store, embedder


def build_reference(width: int, height: int) -> VisualReference:
    return VisualReference(image=Image.new("RGB", (width, height)))


def test_text_queries_are_normalized_and_embedded_once_across_prompts() -> None:
    store, embedder = build_store()
    torch.testing.assert_close(
        store.embed(Prompt.from_class_names(("cat", "dog"))), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    store.embed(Prompt.from_texts({"pet": ("dog", "bird")}))
    store.embed(Prompt.from_class_names(("cat", "dog")))
    assert embedder.embedded_sentences == [["cat", "dog"], ["bird"]]


def test_templates_are_ensembled_per_text_query() -> None:
    store, embedder = build_store(("{}", "a {}"))
    half_sqrt: float = 0.5**0.5
    torch.testing.assert_close(
        store.embed(Prompt.from_class_names(("cat", "dog"))),
        torch.tensor([[half_sqrt, half_sqrt], [half_sqrt, half_sqrt]]),
    )
    assert embedder.embedded_sentences == [["cat", "a cat", "dog", "a dog"]]


def test_visual_query_averages_its_normalized_references() -> None:
    store, embedder = build_store()
    references: tuple[VisualReference, ...] = (build_reference(3, 4), build_reference(4, 3))
    half_sqrt: float = 0.5**0.5
    prompt: Prompt = Prompt({"my mug": (VisualQuery(references),)})
    torch.testing.assert_close(store.embed(prompt), torch.tensor([[half_sqrt, half_sqrt]]))
    store.embed(Prompt({"cup": (VisualQuery((references[1],)),)}))
    assert embedder.embedded_references == list(references)


def test_unused_references_are_released() -> None:
    store, embedder = build_store()
    store.embed(Prompt({"my mug": (VisualQuery((build_reference(5, 5),)),)}))
    embedder.embedded_references.clear()
    gc.collect()
    assert len(store._references.entries) == 0


def test_text_and_visual_queries_keep_query_order() -> None:
    store, embedder = build_store()
    prompt: Prompt = Prompt(
        {
            "pet": (TextQuery("cat"), VisualQuery((build_reference(8, 6),)), TextQuery("dog")),
            "bird": (TextQuery("bird"),),
        }
    )
    half_sqrt: float = 0.5**0.5
    torch.testing.assert_close(
        store.embed(prompt), torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [half_sqrt, half_sqrt]])
    )
    assert embedder.embedded_sentences == [["cat", "dog", "bird"]]


def test_least_recently_used_text_queries_are_evicted() -> None:
    store, embedder = build_store(text_cache_capacity=2)
    store.embed(Prompt.from_class_names(("cat", "dog")))
    store.embed(Prompt.from_class_names(("cat",)))
    store.embed(Prompt.from_class_names(("bird",)))
    store.embed(Prompt.from_class_names(("cat", "dog")))
    assert embedder.embedded_sentences == [["cat", "dog"], ["bird"], ["dog"]]


def test_prompt_larger_than_the_text_cache_is_embedded() -> None:
    store, embedder = build_store(text_cache_capacity=1)
    torch.testing.assert_close(
        store.embed(Prompt.from_class_names(("cat", "dog", "bird"))),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5**0.5, 0.5**0.5]]),
    )
    assert embedder.embedded_sentences == [["cat", "dog", "bird"]]
