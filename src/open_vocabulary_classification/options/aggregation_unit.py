from enum import StrEnum


class AggregationUnit(StrEnum):
    """
    Which embeddings of a class one aggregation step merges, seen as a table of prompts (rows) and templates (columns).

    Attributes
    ----------
    TEMPLATE : str
        The templates of each prompt: one value per prompt remains, e.g. ``"a photo of a dog"`` and
        ``"a drawing of a dog"`` become ``"dog"``.
    PROMPT : str
        The prompts of each template: one value per template remains, e.g. ``"a photo of a dog"`` and
        ``"a photo of a puppy"`` become ``"a photo of a {}"``. Visual queries, which have no template, form one
        group of their own.
    TEMPLATE_AND_PROMPT : str
        Every embedding of the class at once: one value per class remains.
    """

    TEMPLATE = "template"
    PROMPT = "prompt"
    TEMPLATE_AND_PROMPT = "template_and_prompt"
