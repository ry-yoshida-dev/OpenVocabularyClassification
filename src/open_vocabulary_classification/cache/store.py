from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from ..prompt import EmbeddingLayout, TextQuery, TextTemplates, VisualQuery, VisualReference
from .core import EmbeddingCache
from .least_recently_used import LeastRecentlyUsedMapping


@dataclass
class QueryEmbeddingStore:
    """
    Builds the embeddings of a prompt from embeddings cached per sentence and per reference.

    An entry of text sentences averages their normalized embeddings (prompt ensembling when it holds every template);
    a visual query averages the normalized embeddings of its references. Prompts sharing sentences or references
    reuse the cached embeddings, whatever the templates and template embedding; only the assembly, one vectorized
    segment sum over every entry, runs per prompt. At most ``sentence_cache_capacity`` sentences are kept, the least
    recently used dropped first; reference embeddings are held weakly, so they are dropped together with their
    reference. Every cached embedding owns its storage, so evicting it frees its memory. Both caches are thread-safe.

    Attributes
    ----------
    embed_sentences : Callable[[Sequence[str]], torch.Tensor]
        Computes the text embedding of each sentence, shape (S, D).
    embed_references : Callable[[Sequence[VisualReference]], torch.Tensor]
        Computes the image embedding of each reference, shape (R, D).
    sentence_cache_capacity : int
        Maximum number of sentences whose embeddings are kept.
    """

    embed_sentences: Callable[[Sequence[str]], torch.Tensor]
    embed_references: Callable[[Sequence[VisualReference]], torch.Tensor]
    sentence_cache_capacity: int
    _sentences: EmbeddingCache[str, torch.Tensor] = field(init=False, repr=False)
    _references: EmbeddingCache[VisualReference, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sentences = EmbeddingCache(
            compute=self._embed_sentences,
            entries=LeastRecentlyUsedMapping[str, torch.Tensor](self.sentence_cache_capacity),
        )
        self._references = EmbeddingCache(
            compute=self._embed_references, entries=WeakKeyDictionary[VisualReference, torch.Tensor]()
        )

    def embed(self, layout: EmbeddingLayout) -> torch.Tensor:
        """
        Every embedding of a prompt layout.

        Parameters
        ----------
        layout : EmbeddingLayout
            Embeddings of a prompt with text queries, visual queries or both.

        Returns
        -------
        torch.Tensor
            L2-normalized float32 embeddings in embedding-id order, shape (E, D).
        """
        member_sentences: list[str] = []
        sentence_entry_ids: list[int] = []
        member_references: list[VisualReference] = []
        reference_entry_ids: list[int] = []
        for entry_id, entry in enumerate(layout.entries):
            match layout.prompt.queries[entry.query_id]:
                case TextQuery():
                    member_sentences.extend(entry.sentences)
                    sentence_entry_ids.extend([entry_id] * len(entry.sentences))
                case VisualQuery(references=references):
                    member_references.extend(references)
                    reference_entry_ids.extend([entry_id] * len(references))
        member_embeddings: torch.Tensor = torch.stack(
            self._sentences.get(member_sentences) + self._references.get(member_references)
        )
        member_entry_ids: torch.Tensor = torch.tensor(
            sentence_entry_ids + reference_entry_ids, dtype=torch.int64, device=member_embeddings.device
        )
        entry_sums: torch.Tensor = member_embeddings.new_zeros((len(layout), member_embeddings.shape[1]))
        return torch.nn.functional.normalize(entry_sums.index_add_(0, member_entry_ids, member_embeddings), dim=-1)

    def embed_texts(self, texts: Sequence[str], text_templates: TextTemplates) -> torch.Tensor:
        """
        Embedding of text queries, ensembled over the templates.

        Parameters
        ----------
        texts : Sequence[str]
            Text queries; repeated sentences are embedded once.
        text_templates : TextTemplates
            Templates every text query is filled into.

        Returns
        -------
        torch.Tensor
            L2-normalized float32 embeddings in input order, shape (T, D).
        """
        sentences: list[str] = [sentence for text in texts for sentence in text_templates.fill(text)]
        sentence_embeddings: torch.Tensor = torch.stack(self._sentences.get(sentences))
        grouped_embeddings: torch.Tensor = sentence_embeddings.reshape(len(texts), len(text_templates), -1)
        return torch.nn.functional.normalize(grouped_embeddings.sum(dim=1), dim=-1)

    def _embed_sentences(self, sentences: Sequence[str]) -> list[torch.Tensor]:
        return self._normalized_rows(self._checked(self.embed_sentences(sentences), len(sentences)))

    def _embed_references(self, references: Sequence[VisualReference]) -> list[torch.Tensor]:
        return self._normalized_rows(self._checked(self.embed_references(references), len(references)))

    @staticmethod
    def _checked(embeddings: torch.Tensor, expected_count: int) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_count:
            raise ValueError(f"expected embeddings of shape ({expected_count}, D). got {tuple(embeddings.shape)}")
        return embeddings.float()

    @staticmethod
    def _normalized_rows(embeddings: torch.Tensor) -> list[torch.Tensor]:
        return [row.clone() for row in torch.nn.functional.normalize(embeddings, dim=-1)]
