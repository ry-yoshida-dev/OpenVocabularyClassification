from collections.abc import Sequence
from typing import cast

from PIL import Image
from transformers import BatchFeature, SiglipModel, SiglipProcessor

from ..core import SiglipFamilyClassifier
from ..vision_tokens import SiglipVisionTokens


class SiglipClassifier(SiglipFamilyClassifier[SiglipModel, SiglipProcessor]):
    """
    Fixed-resolution SigLIP and SigLIP 2 classifier backed by ``transformers.SiglipModel`` (``model_type`` ``siglip``).

    Images are resized to the square input of the checkpoint without keeping their aspect ratio, so every image has
    the same patch grid.
    """

    def _load_processor(self, weights_path: str) -> SiglipProcessor:
        return SiglipProcessor.from_pretrained(weights_path)

    def _load_model(self, weights_path: str) -> SiglipModel:
        return SiglipModel.from_pretrained(weights_path)

    def _run_vision_tower(self, images: Sequence[Image.Image]) -> SiglipVisionTokens:
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        vision_outputs: object = cast(
            object, self._model.vision_model(pixel_values=self._runtime.to_model_input(image_inputs["pixel_values"]))
        )
        return self._vision_tokens_of(vision_outputs, patch_mask=None)
