from dataclasses import dataclass

from ..options import AggregationTiming, PromptAggregation
from .core import Prompt
from .embedding_entry import EmbeddingEntry
from .entry_groups import EntryGroups
from .queries import TextQuery, VisualQuery
from .text_templates import TextTemplates


@dataclass(frozen=True)
class EmbeddingLayout:
    """
    Embeddings of a prompt the image is compared with, after the aggregation steps at ``EMBEDDING`` timing.

    Every class starts as a table with a row per query and a column per template; a visual query fills one cell.
    The ``EMBEDDING`` steps merge cells into representative embeddings, e.g. ensembling the templates of every
    prompt. Entries are ordered by class. Layouts compare and hash by their fields.

    Attributes
    ----------
    prompt : Prompt
        Prompt the embeddings belong to.
    aggregation : PromptAggregation
        Aggregation whose ``EMBEDDING`` steps the entries went through.
    entries : tuple[EmbeddingEntry, ...]
        Every embedding in embedding-id order.

    Raises
    ------
    ValueError
        If the entries are not ordered by class or a class has none.
    """

    prompt: Prompt
    aggregation: PromptAggregation
    entries: tuple[EmbeddingEntry, ...]

    def __post_init__(self) -> None:
        class_ids: list[int] = [entry.class_id for entry in self.entries]
        if class_ids != sorted(class_ids) or sorted(set(class_ids)) != list(range(len(self.prompt.class_names))):
            raise ValueError("entries must be ordered by class, at least one per class.")

    @classmethod
    def of(cls, prompt: Prompt, text_templates: TextTemplates, aggregation: PromptAggregation) -> "EmbeddingLayout":
        """
        Lay out the embeddings of a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt with text queries, visual queries or both.
        text_templates : TextTemplates
            Templates every text query is filled into.
        aggregation : PromptAggregation
            Aggregation whose ``EMBEDDING`` steps merge the cells.

        Returns
        -------
        EmbeddingLayout
            Entries ordered by class.
        """
        entries: list[EmbeddingEntry] = []
        for query_id, query in enumerate(prompt.queries):
            class_id: int = prompt.query_class_ids[query_id]
            match query:
                case VisualQuery():
                    entries.append(EmbeddingEntry(class_id, query_id, None, (), (query_id,)))
                case TextQuery(text=text):
                    entries.extend(
                        EmbeddingEntry(class_id, query_id, template, (sentence,), ())
                        for sentence, template in zip(text_templates.fill(text), text_templates.templates, strict=True)
                    )
        entries.sort(key=lambda entry: entry.class_id)
        for step in aggregation.steps_at(AggregationTiming.EMBEDDING):
            entries = list(EntryGroups.of(entries, step.unit).merged_entries)
        return cls(prompt=prompt, aggregation=aggregation, entries=tuple(entries))

    def __len__(self) -> int:
        return len(self.entries)
