from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature, CLIPModel, CLIPProcessor, MetaClip2Model
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ...options import ClassifierBackend, ImagePooling
from ..dual_encoder import DualEncoderClassifier


class ClipFamilyClassifier[ModelT: CLIPModel | MetaClip2Model](DualEncoderClassifier[ModelT, CLIPProcessor]):
    """
    Base of CLIP-architecture classifiers backed by Hugging Face ``transformers``.

    The vision tower prepends a class token to the patch tokens and ends in a layer norm and a linear projection into
    the joint space. ``CLASS_TOKEN`` projects the normalized class token, as CLIP was trained; ``PATCH_MEAN``
    normalizes every patch token with the same layer norm and projects their mean. Logits are cosine similarities
    scaled by the learned temperature, without bias, and are turned into scores by a softmax over the classes.
    """

    BACKEND: ClassVar[ClassifierBackend] = ClassifierBackend.CLIP

    def _load_processor(self, weights_path: str) -> CLIPProcessor:
        return CLIPProcessor.from_pretrained(weights_path)

    @property
    def _text_token_limit(self) -> int:
        position_embedding: torch.nn.Embedding = self._model.text_model.embeddings.position_embedding
        return position_embedding.num_embeddings

    def _read_logit_scale(self) -> float:
        return float(self._model.logit_scale.exp().item())

    def _read_logit_bias(self) -> float:
        return 0.0

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        image_inputs: BatchFeature = self._processor.image_processor(images=list(images), return_tensors="pt")
        vision_outputs: BaseModelOutputWithPooling = self._model.vision_model(
            pixel_values=self._runtime.to_model_input(image_inputs["pixel_values"])
        )
        tokens: torch.Tensor | None = vision_outputs.last_hidden_state
        if tokens is None:
            raise RuntimeError("the vision tower returned no token states.")
        image_embeddings: torch.Tensor = self._model.visual_projection(self._pool(tokens))
        return image_embeddings

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Pool the vision tokens before the projection.

        Parameters
        ----------
        tokens : torch.Tensor
            Final encoder states, class token first, before the final layer norm, shape (B, 1 + P, H).

        Returns
        -------
        torch.Tensor
            Normalized pooled states, shape (B, H).

        Raises
        ------
        ValueError
            If ``image_pooling`` is not offered by CLIP.
        """
        post_layernorm: torch.nn.LayerNorm = self._model.vision_model.post_layernorm
        match self.image_pooling:
            case ImagePooling.CLASS_TOKEN:
                class_token: torch.Tensor = post_layernorm(tokens[:, 0])
                return class_token
            case ImagePooling.PATCH_MEAN:
                patch_tokens: torch.Tensor = post_layernorm(tokens[:, 1:])
                return patch_tokens.mean(dim=1)
            case ImagePooling.ATTENTION:
                raise ValueError(f"{type(self).__name__} has no attention pooling head.")

    def _encode_sentences(self, sentences: Sequence[str]) -> torch.Tensor:
        encoding: BatchEncoding = self._tokenize(sentences)
        text_outputs: object = cast(
            object,
            self._model.get_text_features(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]),
        )
        if not isinstance(text_outputs, BaseModelOutputWithPooling) or text_outputs.pooler_output is None:
            raise RuntimeError(f"unexpected text tower output: {type(text_outputs)}")
        return text_outputs.pooler_output
