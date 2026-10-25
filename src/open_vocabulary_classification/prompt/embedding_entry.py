from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingEntry:
    """
    One embedding a classifier compares the image with: a text query filled into templates, or a visual query.

    Attributes
    ----------
    query_id : int
        Query of the prompt the embedding belongs to.
    sentences : tuple[str, ...]
        Sentences whose normalized embeddings are averaged into this one, one per template for an ensembled text
        query, a single sentence otherwise; empty for a visual query, which averages its references instead.
    template : str | None
        Template of the single sentence; ``None`` for an ensembled text query or a visual query.

    Raises
    ------
    ValueError
        If ``query_id`` is negative or a template is given for other than exactly one sentence.
    """

    query_id: int
    sentences: tuple[str, ...]
    template: str | None

    def __post_init__(self) -> None:
        if self.query_id < 0:
            raise ValueError(f"query_id must not be negative. got {self.query_id}")
        if self.template is not None and len(self.sentences) != 1:
            raise ValueError(f"a template belongs to exactly one sentence. got {list(self.sentences)}")
