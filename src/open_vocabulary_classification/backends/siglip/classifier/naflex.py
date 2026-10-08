from collections.abc import Sequence
from typing import cast

import torch
from PIL import Image
from transformers import BatchFeature, Siglip2Model, Siglip2Processor

from ..core import SiglipFamilyClassifier
from ..vision_tokens import SiglipVisionTokens


class Siglip2NaFlexClassifier(SiglipFamilyClassifier[Siglip2Model, Siglip2Processor]):
    """
    SigLIP 2 NaFlex classifier backed by ``transformers.Siglip2Model`` (``model_type`` ``siglip2``).

    Each image is resized to the patch grid closest to its own aspect ratio within the patch budget of the processor,
    and shorter grids are padded within a mini-batch. The patch mask keeps the padding out of both the attention
    pooling and the patch mean.
    """

    def _load_processor(self, weights_path: str) -> Siglip2Processor:
        return Siglip2Processor.from_pretrained(weights_path)

    def _load_model(self, weights_path: str) -> Siglip2Model:
        return Siglip2Model.from_pretrained(weights_path)

    def _run_vision_tower(self, images: Sequence[Image.Image]) -> SiglipVisionTokens:
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        patch_mask: torch.Tensor = self._runtime.to_device(image_inputs["pixel_attention_mask"])
        vision_outputs: object = cast(
            object,
            self._model.vision_model(
                pixel_values=self._runtime.to_model_input(image_inputs["pixel_values"]),
                pixel_attention_mask=patch_mask,
                spatial_shapes=self._runtime.to_device(image_inputs["spatial_shapes"]),
            ),
        )
        return self._vision_tokens_of(vision_outputs, patch_mask=patch_mask.bool())
