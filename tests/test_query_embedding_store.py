import gc
import threading
from collections.abc import Sequence

import torch
from PIL import Image

from open_vocabulary_classification import (
    AggregationMethod,
    AggregationStep,
    AggregationTiming,
    AggregationUnit,
    EmbeddingLayout,
    Prompt,
    PromptAggregation,
    TextQuery,
    TextTemplates,
    VisualQuery,
    VisualReference,
)
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


PLAIN_TEMPLATES: TextTemplates = TextTemplates(("{}",))
ARTICLE_TEMPLATES: TextTemplates = TextTemplates(("{}", "a {}"))


def build_store(sentence_cache_capacity: int = 16) -> tuple[QueryEmbeddingStore, CountingEmbedder]:
    embedder: CountingEmbedder = CountingEmbedder()
    store: QueryEmbeddingStore = QueryEmbeddingStore(
        embed_sentences=embedder.embed_sentences,
        embed_references=embedder.embed_references,
        sentence_cache_capacity=sentence_cache_capacity,
    )
    return store, embedder


ENSEMBLED: PromptAggregation = PromptAggregation()
PER_SENTENCE: PromptAggregation = PromptAggregation.of(
    [
        AggregationStep(AggregationUnit.TEMPLATE, AggregationTiming.SIMILARITY, AggregationMethod.MAX),
        AggregationStep(AggregationUnit.PROMPT, AggregationTiming.SIMILARITY, AggregationMethod.MAX),
    ]
)
REPRESENTATIVE: PromptAggregation = PromptAggregation.of(
    [AggregationStep(AggregationUnit.TEMPLATE_AND_PROMPT, AggregationTiming.EMBEDDING, AggregationMethod.MEAN)]
)


def layout_of(
    prompt: Prompt,
    templates: TextTemplates = PLAIN_TEMPLATES,
    prompt_aggregation: PromptAggregation = ENSEMBLED,
) -> EmbeddingLayout:
    return EmbeddingLayout.of(prompt, templates, prompt_aggregation)


def build_reference(width: int, height: int) -> VisualReference:
    return VisualReference(image=Image.new("RGB", (width, height)))


def test_text_queries_are_normalized_and_embedded_once_across_prompts() -> None:
    store, embedder = build_store()
    torch.testing.assert_close(
        store.embed(layout_of(Prompt.from_class_names(("cat", "dog")))), torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    store.embed(layout_of(Prompt.from_texts({"pet": ("dog", "bird")})))
    store.embed(layout_of(Prompt.from_class_names(("cat", "dog"))))
    assert embedder.embedded_sentences == [["cat", "dog"], ["bird"]]


def test_templates_are_ensembled_per_text_query() -> None:
    store, embedder = build_store()
    half_sqrt: float = 0.5**0.5
    torch.testing.assert_close(
        store.embed(layout_of(Prompt.from_class_names(("cat", "dog")), ARTICLE_TEMPLATES)),
        torch.tensor([[half_sqrt, half_sqrt], [half_sqrt, half_sqrt]]),
    )
    assert embedder.embedded_sentences == [["cat", "a cat", "dog", "a dog"]]


def test_visual_query_averages_its_normalized_references() -> None:
    store, embedder = build_store()
    references: tuple[VisualReference, ...] = (build_reference(3, 4), build_reference(4, 3))
    half_sqrt: float = 0.5**0.5
    prompt: Prompt = Prompt({"my mug": (VisualQuery(references),)})
    torch.testing.assert_close(store.embed(layout_of(prompt)), torch.tensor([[half_sqrt, half_sqrt]]))
    store.embed(layout_of(Prompt({"cup": (VisualQuery((references[1],)),)})))
    assert embedder.embedded_references == list(references)


def test_unused_references_are_released() -> None:
    store, embedder = build_store()
    store.embed(layout_of(Prompt({"my mug": (VisualQuery((build_reference(5, 5),)),)})))
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
        store.embed(layout_of(prompt)), torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [half_sqrt, half_sqrt]])
    )
    assert embedder.embedded_sentences == [["cat", "dog", "bird"]]


def test_least_recently_used_sentences_are_evicted() -> None:
    store, embedder = build_store(sentence_cache_capacity=2)
    store.embed(layout_of(Prompt.from_class_names(("cat", "dog"))))
    store.embed(layout_of(Prompt.from_class_names(("cat",))))
    store.embed(layout_of(Prompt.from_class_names(("bird",))))
    store.embed(layout_of(Prompt.from_class_names(("cat", "dog"))))
    assert embedder.embedded_sentences == [["cat", "dog"], ["bird"], ["dog"]]


def test_prompt_larger_than_the_sentence_cache_is_embedded() -> None:
    store, embedder = build_store(sentence_cache_capacity=1)
    torch.testing.assert_close(
        store.embed(layout_of(Prompt.from_class_names(("cat", "dog", "bird")))),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5**0.5, 0.5**0.5]]),
    )
    assert embedder.embedded_sentences == [["cat", "dog", "bird"]]


def test_embed_texts_shares_the_cache_with_prompts() -> None:
    store, embedder = build_store()
    torch.testing.assert_close(
        store.embed_texts(["dog", "cat", "dog"], PLAIN_TEMPLATES), torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    )
    store.embed(layout_of(Prompt.from_class_names(("cat", "dog", "bird"))))
    assert embedder.embedded_sentences == [["dog", "cat"], ["bird"]]


def test_concurrent_prompts_embed_each_sentence_once() -> None:
    store, embedder = build_store()
    barrier: threading.Barrier = threading.Barrier(8)

    def embed() -> None:
        barrier.wait()
        store.embed(layout_of(Prompt.from_class_names(("cat", "dog", "bird"))))

    threads: list[threading.Thread] = [threading.Thread(target=embed) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert embedder.embedded_sentences == [["cat", "dog", "bird"]]


def test_per_sentence_keeps_one_embedding_per_template() -> None:
    store, _ = build_store()
    layout: EmbeddingLayout = layout_of(Prompt.from_class_names(("cat", "dog")), ARTICLE_TEMPLATES, PER_SENTENCE)
    torch.testing.assert_close(store.embed(layout), torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]))
    assert [entry.template for entry in layout.entries] == ["{}", "a {}", "{}", "a {}"]
    assert [entry.query_id for entry in layout.entries] == [0, 0, 1, 1]


def test_representative_embedding_averages_sentences_and_visual_queries_as_members() -> None:
    store, _ = build_store()
    references: tuple[VisualReference, ...] = (build_reference(3, 0), build_reference(0, 3))
    prompt: Prompt = Prompt({"pet": (TextQuery("cat"), TextQuery("dog"), VisualQuery(references))})
    half_sqrt: float = 0.5**0.5
    visual: list[float] = [half_sqrt, half_sqrt]
    expected: torch.Tensor = torch.nn.functional.normalize(
        torch.tensor([1.0, 0.0]) + torch.tensor([0.0, 1.0]) + torch.tensor(visual), dim=-1
    )
    torch.testing.assert_close(store.embed(layout_of(prompt, PLAIN_TEMPLATES, REPRESENTATIVE)), expected[None])


def test_sentences_are_shared_between_aggregations() -> None:
    store, embedder = build_store()
    prompt: Prompt = Prompt.from_class_names(("cat",))
    store.embed(layout_of(prompt, ARTICLE_TEMPLATES, PER_SENTENCE))
    store.embed(layout_of(prompt, ARTICLE_TEMPLATES))
    store.embed_texts(["cat"], ARTICLE_TEMPLATES)
    assert embedder.embedded_sentences == [["cat", "a cat"]]
