from transformers import MetaClip2Model

from ..core import ClipFamilyClassifier


class MetaClip2Classifier(ClipFamilyClassifier[MetaClip2Model]):
    """
    MetaCLIP 2 classifier backed by ``transformers.MetaClip2Model`` (``model_type`` ``metaclip_2``).

    The CLIP architecture trained on worldwide, multilingual data with an XLM-RoBERTa tokenizer, so text queries and
    templates may be written in many languages.
    """

    def _load_model(self, weights_path: str) -> MetaClip2Model:
        return MetaClip2Model.from_pretrained(weights_path)
