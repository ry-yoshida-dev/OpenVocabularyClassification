from transformers import CLIPModel

from ..core import ClipFamilyClassifier


class ClipClassifier(ClipFamilyClassifier[CLIPModel]):
    """
    CLIP classifier backed by ``transformers.CLIPModel`` (``model_type`` ``clip``).

    Loads OpenAI CLIP, OpenCLIP weights converted to ``transformers`` (e.g. LAION, DataComp) and MetaCLIP.
    """

    def _load_model(self, weights_path: str) -> CLIPModel:
        return CLIPModel.from_pretrained(weights_path)
