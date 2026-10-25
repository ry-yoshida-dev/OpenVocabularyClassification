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
    The ``EMBEDDING`` steps merge the cells step by step into representative embeddings, e.g. ensembling the
    templates of every prompt; every step averages the normalized embeddings of its groups and normalizes again.
    Entries and cells are ordered by class. Layouts compare and hash by their fields.

    Attributes
    ----------
    prompt : Prompt
        Prompt the embeddings belong to.
    aggregation : PromptAggregation
        Aggregation whose ``EMBEDDING`` steps the entries went through.
    cells : tuple[EmbeddingEntry, ...]
        Every cell of the class tables, a sentence or a visual query each.
    embedding_group_ids : tuple[tuple[int, ...], ...]
        Group of every input entry of each ``EMBEDDING`` step, in step order; the first step groups the cells.
    entries : tuple[EmbeddingEntry, ...]
        Every embedding in embedding-id order, the groups of the last step.

    Raises
    ------
    ValueError
        If the entries are not ordered by class or a class has none, a cell holds other than exactly one sentence or
        visual query, or the group ids do not chain the cells into the entries.
    """

    prompt: Prompt
    aggregation: PromptAggregation
    cells: tuple[EmbeddingEntry, ...]
    embedding_group_ids: tuple[tuple[int, ...], ...]
    entries: tuple[EmbeddingEntry, ...]

    def __post_init__(self) -> None:
        class_ids: list[int] = [entry.class_id for entry in self.entries]
        if class_ids != sorted(class_ids) or sorted(set(class_ids)) != list(range(len(self.prompt.class_names))):
            raise ValueError("entries must be ordered by class, at least one per class.")
        if any(len(cell.sentences) + len(cell.visual_query_ids) != 1 for cell in self.cells):
            raise ValueError("every cell must hold exactly one sentence or visual query.")
        input_count: int = len(self.cells)
        for group_ids in self.embedding_group_ids:
            if len(group_ids) != input_count or sorted(set(group_ids)) != list(range(len(set(group_ids)))):
                raise ValueError(f"every embedding step must number groups of its {input_count} inputs from 0.")
            input_count = len(set(group_ids))
        if input_count != len(self.entries):
            raise ValueError(f"the embedding steps must leave one group per entry. got {input_count} groups")

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
        cells: tuple[EmbeddingEntry, ...] = tuple(entries)
        embedding_group_ids: list[tuple[int, ...]] = []
        for step in aggregation.steps_at(AggregationTiming.EMBEDDING):
            groups: EntryGroups = EntryGroups.of(entries, step.unit)
            embedding_group_ids.append(tuple(int(group_id) for group_id in groups.group_ids.tolist()))
            entries = list(groups.merged_entries)
        return cls(
            prompt=prompt,
            aggregation=aggregation,
            cells=cells,
            embedding_group_ids=tuple(embedding_group_ids),
            entries=tuple(entries),
        )

    def __len__(self) -> int:
        return len(self.entries)
