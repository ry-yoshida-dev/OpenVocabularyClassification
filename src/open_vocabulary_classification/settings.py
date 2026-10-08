from dataclasses import dataclass
from typing import TYPE_CHECKING

from .options import ClassifierBackend, Device, ImagePooling
from .prompt import TextTemplates

if TYPE_CHECKING:
    from .classifier import OpenVocabularyClassifier


@dataclass(frozen=True, kw_only=True)
class ClassifierSettings:
    """
    Everything needed to load and run a classifier, identical for every backend.

    The fields map one-to-one onto the ``classifier`` section of the YAML presets in this package,
    so a preset can be turned into settings by any dataclass builder (enums are written by value).

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
    is_half_precision_enabled : bool
        Whether to run the model in float16; GPU only.

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
    is_half_precision_enabled: bool = False

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
                from .backends.clip import ClipArchitecture

                return ClipArchitecture.of_checkpoint(self.weights_path).build(self)
            case ClassifierBackend.SIGLIP:
                from .backends.siglip import SiglipArchitecture

                return SiglipArchitecture.of_checkpoint(self.weights_path).build(self)
