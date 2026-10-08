from dataclasses import dataclass
from string import Formatter


@dataclass(frozen=True)
class TextTemplates:
    """
    Sentences placing a text query in context, ensembled into one text embedding (prompt ensembling).

    Every text query is filled into every template, e.g. ``"a photo of a {}."`` turns ``"dog"`` into
    ``"a photo of a dog."``; the L2-normalized embeddings of the filled sentences are averaged and normalized again.
    ``("{}",)`` embeds the text query as it is.

    Attributes
    ----------
    templates : tuple[str, ...]
        Templates with exactly one ``{}`` placeholder each; literal braces are written ``{{`` and ``}}``.

    Raises
    ------
    ValueError
        If there is no template, a template is repeated, is malformed, or does not have exactly one bare ``{}``.
    """

    templates: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("at least one text template is needed.")
        if len(set(self.templates)) != len(self.templates):
            raise ValueError(f"text templates must be unique. got {list(self.templates)}")
        for template in self.templates:
            self._validate_template(template)

    def __len__(self) -> int:
        return len(self.templates)

    def fill(self, text: str) -> tuple[str, ...]:
        """
        Fill a text query into every template.

        Parameters
        ----------
        text : str
            Text query, e.g. ``"dog"``.

        Returns
        -------
        tuple[str, ...]
            One sentence per template, in template order.
        """
        return tuple(template.format(text) for template in self.templates)

    @staticmethod
    def _validate_template(template: str) -> None:
        replacement_fields: list[tuple[str, str | None, str | None]] = [
            (field_name, format_spec, conversion)
            for _, field_name, format_spec, conversion in Formatter().parse(template)
            if field_name is not None
        ]
        if replacement_fields != [("", "", None)]:
            raise ValueError(f"text templates need exactly one bare '{{}}' placeholder. got {template!r}")
