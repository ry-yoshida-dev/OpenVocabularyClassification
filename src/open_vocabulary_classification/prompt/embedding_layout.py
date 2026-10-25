from dataclasses import dataclass, field

import numpy as np

from ..array_types import IntArray
from ..options import TemplateCombination
from .core import Prompt
from .embedding_entry import EmbeddingEntry
from .queries import TextQuery, VisualQuery
from .text_templates import TextTemplates


@dataclass(frozen=True)
class EmbeddingLayout:
    """
    Embeddings of a prompt the image is compared with, in embedding-id order, grouped by query.

    A visual query has one embedding. A text query has one embedding ensembling its sentences (``ENSEMBLE``) or one
    per sentence in template order (every other ``TemplateCombination``). Layouts compare and hash by their prompt and
    entries.

    Attributes
    ----------
    prompt : Prompt
        Prompt the embeddings belong to.
    entries : tuple[EmbeddingEntry, ...]
        Every embedding in embedding-id order.
    entry_query_ids : IntArray
        Query of every embedding, non-decreasing, shape (E,); read-only.

    Raises
    ------
    ValueError
        If the entries are not grouped by query in query-id order with at least one entry per query, or an entry of
        a text query has no sentence or one of a visual query has some.
    """

    prompt: Prompt
    entries: tuple[EmbeddingEntry, ...]
    entry_query_ids: IntArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        query_ids: list[int] = [entry.query_id for entry in self.entries]
        expected_ids: list[int] = sorted(query_ids)
        if query_ids != expected_ids or sorted(set(query_ids)) != list(range(len(self.prompt.queries))):
            raise ValueError("entries must be grouped by query in query-id order, at least one per query.")
        for entry in self.entries:
            if isinstance(self.prompt.queries[entry.query_id], VisualQuery) == bool(entry.sentences):
                raise ValueError("text queries need sentences and visual queries none.")
        entry_query_ids: IntArray = np.array(query_ids, dtype=np.int64)
        entry_query_ids.setflags(write=False)
        object.__setattr__(self, "entry_query_ids", entry_query_ids)

    @classmethod
    def of(
        cls, prompt: Prompt, text_templates: TextTemplates, template_combination: TemplateCombination
    ) -> "EmbeddingLayout":
        """
        Lay out the embeddings of a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt with text queries, visual queries or both.
        text_templates : TextTemplates
            Templates every text query is filled into.
        template_combination : TemplateCombination
            How the sentences of a text query are combined; only ``ENSEMBLE`` merges them into one embedding.

        Returns
        -------
        EmbeddingLayout
            Entries of every query in query-id order.
        """
        entries: list[EmbeddingEntry] = []
        for query_id, query in enumerate(prompt.queries):
            match query:
                case VisualQuery():
                    entries.append(EmbeddingEntry(query_id=query_id, sentences=(), template=None))
                case TextQuery(text=text):
                    sentences: tuple[str, ...] = text_templates.fill(text)
                    if template_combination.is_ensembled:
                        entries.append(EmbeddingEntry(query_id=query_id, sentences=sentences, template=None))
                    else:
                        entries.extend(
                            EmbeddingEntry(query_id=query_id, sentences=(sentence,), template=template)
                            for sentence, template in zip(sentences, text_templates.templates, strict=True)
                        )
        return cls(prompt=prompt, entries=tuple(entries))

    def __len__(self) -> int:
        return len(self.entries)
