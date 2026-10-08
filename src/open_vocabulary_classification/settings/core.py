from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from ..options import ClassifierBackend, Device, ImagePooling, Precision
from ..prompt import TextTemplates
from .config_values import ConfigValues

if TYPE_CHECKING:
    from ..classifier import OpenVocabularyClassifier


@dataclass(frozen=True, kw_only=True)
class ClassifierSettings:
    """
    Everything needed to load and run a classifier, identical for every backend.

    The fields map one-to-one onto the ``classifier`` section of the YAML presets in this package (enums written by
    value), read by ``from_mapping`` or ``PresetCatalog``.

    Attributes
    ----------
    backend : ClassifierBackend
        Classifier family able to load ``weights_path``.
    weights_path : str
        Hugging Face Hub model id or local checkpoint directory.
    image_pooling : ImagePooling
        How the vision tokens are pooled into the image embedding; one of ``backend.supported_image_poolings``.
    text_templates : tuple[str, ...]
        Templates with one ``{}`` each, filled with every text query and ensembled; ``("{}",)`` uses the bare query.
    batch_size : int
        Number of images per forward pass.
    device : Device
        Device to run on.
    precision : Precision
        Floating-point type of the model weights; ``float16`` needs a GPU.

    Raises
    ------
    ValueError
        If ``weights_path`` is blank, ``batch_size`` is not positive, the backend does not offer ``image_pooling``,
        or a text template is invalid.
    """

    backend: ClassifierBackend
    weights_path: str
    image_pooling: ImagePooling
    text_templates: tuple[str, ...] = ("{}",)
    batch_size: int = 8
    device: Device = Device.AUTO
    precision: Precision = Precision.FLOAT32

    def __post_init__(self) -> None:
        if not self.weights_path.strip():
            raise ValueError("weights_path must not be blank.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive. got {self.batch_size}")
        if self.image_pooling not in self.backend.supported_image_poolings:
            raise ValueError(
                f"{self.backend} does not offer {self.image_pooling} pooling. "
                + f"supported: {sorted(pooling.value for pooling in self.backend.supported_image_poolings)}"
            )
        TextTemplates(self.text_templates)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ClassifierSettings":
        """
        Build settings from plain values, e.g. the ``classifier`` section of a parsed YAML or JSON document.

        Enums are given by value (``backend: siglip``) and ``text_templates`` as a list; omitted optional fields take
        their defaults.

        Parameters
        ----------
        values : Mapping[str, object]
            Field values keyed by field name.

        Returns
        -------
        ClassifierSettings
            Validated settings.

        Raises
        ------
        KeyError
            If a required field is missing or an unknown field is given.
        TypeError
            If a value has the wrong type.
        ValueError
            If an enum value is unknown or the settings are invalid.
        """
        config_values: ConfigValues = ConfigValues(values)
        config_values.validate_keys(frozenset(field.name for field in fields(cls)))
        required: ClassifierSettings = cls(
            backend=config_values.member("backend", ClassifierBackend),
            weights_path=config_values.text("weights_path"),
            image_pooling=config_values.member("image_pooling", ImagePooling),
        )
        return cls(
            backend=required.backend,
            weights_path=required.weights_path,
            image_pooling=required.image_pooling,
            text_templates=config_values.texts("text_templates", required.text_templates),
            batch_size=config_values.integer("batch_size", required.batch_size),
            device=config_values.member("device", Device, required.device),
            precision=config_values.member("precision", Precision, required.precision),
        )

    def build(self) -> "OpenVocabularyClassifier":
        """
        Load the classifier of ``backend`` with these settings.

        Only the sub-package of ``backend`` is imported. The ``model_type`` of the checkpoint configuration selects
        the architecture within the family (CLIP or MetaCLIP 2, fixed-resolution SigLIP or SigLIP 2 NaFlex).

        Returns
        -------
        OpenVocabularyClassifier
            Loaded classifier.

        Raises
        ------
        ValueError
            If the checkpoint has an architecture of another backend.
        """
        match self.backend:
            case ClassifierBackend.CLIP:
                from ..backends.clip import ClipArchitecture

                return ClipArchitecture.of_checkpoint(self.weights_path).build(self)
            case ClassifierBackend.SIGLIP:
                from ..backends.siglip import SiglipArchitecture

                return SiglipArchitecture.of_checkpoint(self.weights_path).build(self)
