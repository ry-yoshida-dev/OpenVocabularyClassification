from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

import torch

from ..prompt import EmbeddingLayout, TextTemplates, VisualQuery, VisualReference
from .core import EmbeddingCache
from .least_recently_used import LeastRecentlyUsedMapping


@dataclass
class QueryEmbeddingStore:
    """
    Builds the embeddings of a prompt from embeddings cached per sentence and per reference.

    Every cell of the layout is a normalized sentence embedding or a visual query averaging the normalized embeddings
    of its references; every ``EMBEDDING`` step of the layout then averages its groups and normalizes again. Prompts
    sharing sentences or references reuse the cached embeddings, whatever the templates and aggregation; only the
    assembly, one vectorized segment sum over every entry, runs per prompt. At most ``sentence_cache_capacity``
    sentences are kept, the least recently used dropped first; reference embeddings are held weakly, so they are
    dropped together with their reference. Every cached embedding owns its storage, so evicting it frees its memory.
    Both caches are thread-safe.

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
        sentences: list[str] = [sentence for cell in layout.cells for sentence in cell.sentences]
        sentence_embeddings: list[torch.Tensor] = self._sentences.get(sentences)
        cell_embeddings: list[torch.Tensor] = []
        sentence_index: int = 0
        for cell in layout.cells:
            if cell.sentences:
                cell_embeddings.append(sentence_embeddings[sentence_index])
                sentence_index += 1
                continue
            query: object = layout.prompt.queries[cell.visual_query_ids[0]]
            if not isinstance(query, VisualQuery):
                raise TypeError(f"the query of a visual cell must be a VisualQuery. got {query!r}")
            cell_embeddings.append(self._visual_query_embedding(query))
        embeddings: torch.Tensor = torch.stack(cell_embeddings)
        for group_ids in layout.embedding_group_ids:
            group_index: torch.Tensor = torch.tensor(group_ids, dtype=torch.int64, device=embeddings.device)
            group_sums: torch.Tensor = embeddings.new_zeros((max(group_ids) + 1, embeddings.shape[1]))
            embeddings = torch.nn.functional.normalize(group_sums.index_add_(0, group_index, embeddings), dim=-1)
        return embeddings

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

    def _visual_query_embedding(self, query: VisualQuery) -> torch.Tensor:
        reference_embeddings: torch.Tensor = torch.stack(self._references.get(query.references))
        return torch.nn.functional.normalize(reference_embeddings.mean(dim=0), dim=-1)

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
