from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar, cast

import torch
from PIL import Image
from transformers import BatchEncoding, Siglip2Model, Siglip2Processor, SiglipModel, SiglipProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ...options import ClassifierBackend, ImagePooling
from ...settings import ClassifierSettings
from ..dual_encoder import DualEncoderClassifier
from .uniform_attention_pooling import UniformAttentionPooling
from .vision_tokens import SiglipVisionTokens


class SiglipFamilyClassifier[
    ModelT: SiglipModel | Siglip2Model,
    ProcessorT: SiglipProcessor | Siglip2Processor,
](DualEncoderClassifier[ModelT, ProcessorT]):
    """
    Base of SigLIP and SigLIP 2 classifiers backed by Hugging Face ``transformers``.

    The vision tower has no class token; an attention pooling head (MAP) reduces the normalized patch tokens to the
    image embedding. ``ATTENTION`` uses that head as SigLIP was trained; ``PATCH_MEAN`` averages the patch tokens and
    sends the mean through the head with uniform attention weights (``UniformAttentionPooling``). Text is lowercased,
    as SigLIP 2 was trained and the SigLIP tokenizer does, padded to the full text length and encoded without an
    attention mask, as the text towers were trained. Logits are scaled cosine similarities plus a learned bias, and
    are turned into scores by a sigmoid per class.

    Subclasses run the vision tower on the preprocessed images of their architecture.
    """

    BACKEND: ClassVar[ClassifierBackend] = ClassifierBackend.SIGLIP

    def __init__(self, settings: ClassifierSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : ClassifierSettings
            Checkpoint, pooling, text templates, batching and device.

        Raises
        ------
        ValueError
            If the vision tower of the checkpoint has no attention pooling head.
        """
        super().__init__(settings)
        if not self._model.vision_model.use_head:
            raise ValueError(f"{settings.weights_path} has no attention pooling head to embed images with.")

    @abstractmethod
    def _run_vision_tower(self, images: Sequence[Image.Image]) -> SiglipVisionTokens:
        """
        Preprocess RGB images and run the vision tower on them.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.

        Returns
        -------
        SiglipVisionTokens
            Normalized patch tokens, attention-pooled embeddings and the patch mask.
        """

    @property
    def _text_token_limit(self) -> int:
        position_embedding: torch.nn.Embedding = self._model.text_model.embeddings.position_embedding
        return position_embedding.num_embeddings

    def _read_logit_scale(self) -> float:
        return float(self._model.logit_scale.exp().item())

    def _read_logit_bias(self) -> float:
        return float(self._model.logit_bias.item())

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        vision_tokens: SiglipVisionTokens = self._run_vision_tower(images)
        match self.image_pooling:
            case ImagePooling.ATTENTION:
                return vision_tokens.attention_pooled
            case ImagePooling.PATCH_MEAN:
                return UniformAttentionPooling.pool(
                    self._model.vision_model.head, vision_tokens.tokens, vision_tokens.patch_mask
                )
            case ImagePooling.CLASS_TOKEN:
                raise ValueError(f"{type(self).__name__} has no class token.")

    def _encode_sentences(self, sentences: Sequence[str]) -> torch.Tensor:
        encoding: BatchEncoding = self._tokenize([sentence.lower() for sentence in sentences])
        text_outputs: object = cast(object, self._model.get_text_features(input_ids=encoding["input_ids"]))
        if not isinstance(text_outputs, BaseModelOutputWithPooling) or text_outputs.pooler_output is None:
            raise RuntimeError(f"unexpected text tower output: {type(text_outputs)}")
        return text_outputs.pooler_output

    @staticmethod
    def _vision_tokens_of(vision_outputs: object, patch_mask: torch.Tensor | None) -> SiglipVisionTokens:
        """
        Read the token states and attention-pooled embedding of a vision tower output.

        Parameters
        ----------
        vision_outputs : object
            Output of the vision tower.
        patch_mask : torch.Tensor | None
            Whether each token is an image patch, shape (B, P); ``None`` when every token is a patch.

        Returns
        -------
        SiglipVisionTokens
            Tokens, attention-pooled embeddings and patch mask.

        Raises
        ------
        RuntimeError
            If the output lacks the token states or the pooled embedding.
        """
        if (
            not isinstance(vision_outputs, BaseModelOutputWithPooling)
            or vision_outputs.last_hidden_state is None
            or vision_outputs.pooler_output is None
        ):
            raise RuntimeError(f"unexpected vision tower output: {type(vision_outputs)}")
        return SiglipVisionTokens(
            tokens=vision_outputs.last_hidden_state,
            attention_pooled=vision_outputs.pooler_output,
            patch_mask=patch_mask,
        )
