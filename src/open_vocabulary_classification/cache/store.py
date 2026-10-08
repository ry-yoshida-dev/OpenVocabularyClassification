from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from ..prompt import Prompt, PromptQuery, TextQuery, TextTemplates, VisualQuery, VisualReference
from .core import EmbeddingCache
from .least_recently_used import LeastRecentlyUsedMapping


@dataclass
class QueryEmbeddingStore:
    """
    Builds the query embeddings of a prompt from embeddings cached per text query and per reference.

    A text query is filled into every template and the normalized sentence embeddings are averaged (prompt
    ensembling); a visual query averages the normalized embeddings of its references. Prompts sharing text queries
    or references reuse the cached embeddings; only the cheap assembly runs per prompt. At most
    ``text_cache_capacity`` text queries are kept, the least recently used dropped first; reference embeddings are
    held weakly, so they are dropped together with their reference. Both caches are thread-safe.

    Attributes
    ----------
    embed_sentences : Callable[[Sequence[str]], torch.Tensor]
        Computes the text embedding of each sentence, shape (S, D).
    embed_references : Callable[[Sequence[VisualReference]], torch.Tensor]
        Computes the image embedding of each reference, shape (R, D).
    text_templates : TextTemplates
        Templates every text query is filled into.
    text_cache_capacity : int
        Maximum number of text queries whose embeddings are kept.
    """

    embed_sentences: Callable[[Sequence[str]], torch.Tensor]
    embed_references: Callable[[Sequence[VisualReference]], torch.Tensor]
    text_templates: TextTemplates
    text_cache_capacity: int
    _texts: EmbeddingCache[str, torch.Tensor] = field(init=False, repr=False)
    _references: EmbeddingCache[VisualReference, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._texts = EmbeddingCache(
            compute=self._embed_text_queries,
            entries=LeastRecentlyUsedMapping[str, torch.Tensor](self.text_cache_capacity),
        )
        self._references = EmbeddingCache(
            compute=self._embed_references, entries=WeakKeyDictionary[VisualReference, torch.Tensor]()
        )

    def embed(self, prompt: Prompt) -> torch.Tensor:
        """
        Embedding of every query of a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt with text queries, visual queries or both.

        Returns
        -------
        torch.Tensor
            L2-normalized float32 embeddings in query-id order, shape (Q, D).
        """
        texts: list[str] = [query.text for query in prompt.queries if isinstance(query, TextQuery)]
        text_embeddings: dict[str, torch.Tensor] = dict(zip(texts, self._texts.get(texts), strict=True))
        return torch.stack([self._embed_query(query, text_embeddings) for query in prompt.queries])

    def embed_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Embedding of text queries, ensembled over the templates exactly as the text queries of a prompt.

        Parameters
        ----------
        texts : Sequence[str]
            Text queries; repeated texts are embedded once.

        Returns
        -------
        torch.Tensor
            L2-normalized float32 embeddings in input order, shape (T, D).
        """
        return torch.stack(self._texts.get(texts))

    def _embed_query(self, query: PromptQuery, text_embeddings: Mapping[str, torch.Tensor]) -> torch.Tensor:
        match query:
            case TextQuery(text=text):
                return text_embeddings[text]
            case VisualQuery(references=references):
                return self._normalized_mean(torch.stack(self._references.get(references)))

    def _embed_text_queries(self, texts: Sequence[str]) -> list[torch.Tensor]:
        sentences: list[str] = [sentence for text in texts for sentence in self.text_templates.fill(text)]
        sentence_embeddings: torch.Tensor = self._checked(self.embed_sentences(sentences), len(sentences))
        grouped_embeddings: torch.Tensor = sentence_embeddings.reshape(len(texts), len(self.text_templates), -1)
        return [self._normalized_mean(embeddings) for embeddings in grouped_embeddings]

    def _embed_references(self, references: Sequence[VisualReference]) -> list[torch.Tensor]:
        reference_embeddings: torch.Tensor = self._checked(self.embed_references(references), len(references))
        return list(torch.nn.functional.normalize(reference_embeddings, dim=-1))

    @staticmethod
    def _checked(embeddings: torch.Tensor, expected_count: int) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_count:
            raise ValueError(f"expected embeddings of shape ({expected_count}, D). got {tuple(embeddings.shape)}")
        return embeddings.float()

    @staticmethod
    def _normalized_mean(embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings: torch.Tensor = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        return torch.nn.functional.normalize(normalized_embeddings.mean(dim=0), dim=-1)
