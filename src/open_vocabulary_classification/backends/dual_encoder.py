from abc import abstractmethod
from collections.abc import Sequence
from itertools import batched
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from transformers import BatchEncoding, PreTrainedModel, ProcessorMixin

from ..array_types import FloatArray
from ..cache import QueryEmbeddingStore
from ..classifier import OpenVocabularyClassifier
from ..prompt import Prompt, TextTemplates, VisualReference
from ..runtime import TorchRuntime
from ..settings import ClassifierSettings


class DualEncoderClassifier[ModelT: PreTrainedModel, ProcessorT: ProcessorMixin](OpenVocabularyClassifier):
    """
    Base of classifiers backed by a Hugging Face ``transformers`` dual encoder, whose image and text towers embed
    into one space where the scaled cosine similarity (plus a bias for SigLIP) is the logit.

    Text queries are embedded through the text templates and cached per query; visual queries are embedded by the
    image tower with the configured pooling and cached per reference. The logit scale and bias are read from the model
    once after loading, before the model is cast to ``settings.precision``, so a half-precision model scores with the
    exact learned values. Subclasses load the model and processor, encode sentences and pooled images, and read the
    logit scale and bias of their model.
    """

    TEXT_BATCH_SIZE: ClassVar[int] = 256
    TEXT_CACHE_CAPACITY: ClassVar[int] = 4096

    def __init__(self, settings: ClassifierSettings) -> None:
        """
        Load the processor and model.

        Parameters
        ----------
        settings : ClassifierSettings
            Checkpoint, pooling, text templates, batching and device.
        """
        super().__init__(settings)
        self._runtime: TorchRuntime = TorchRuntime(settings)
        self._processor: ProcessorT = self._load_processor(settings.weights_path)
        self._model: ModelT = self._load_model(settings.weights_path)
        with torch.inference_mode():
            self._logit_scale: float = self._read_logit_scale()
            self._logit_bias: float = self._read_logit_bias()
        self._runtime.prepare_model(self._model)
        self._query_embeddings: QueryEmbeddingStore = QueryEmbeddingStore(
            embed_sentences=self._embed_sentences,
            embed_references=self._embed_references,
            text_templates=TextTemplates(settings.text_templates),
            text_cache_capacity=self.TEXT_CACHE_CAPACITY,
        )

    @abstractmethod
    def _load_processor(self, weights_path: str) -> ProcessorT:
        """
        Load the processor of the checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        ProcessorT
            Loaded processor.
        """

    @abstractmethod
    def _load_model(self, weights_path: str) -> ModelT:
        """
        Load the dual encoder of the checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Returns
        -------
        ModelT
            Loaded model.
        """

    @property
    @abstractmethod
    def _text_token_limit(self) -> int:
        """
        Number of tokens the text tower reads.

        Returns
        -------
        int
            Text positions of the model, padding included.
        """

    @abstractmethod
    def _read_logit_scale(self) -> float:
        """
        Read the factor turning cosine similarities into logits from the loaded model.

        Called once after loading, before the model is moved and cast to the runtime precision, in inference mode.

        Returns
        -------
        float
            Exponentiated learned temperature of the model.
        """

    @abstractmethod
    def _read_logit_bias(self) -> float:
        """
        Read the offset added to the scaled similarities from the loaded model.

        Called once after loading, before the model is moved and cast to the runtime precision, in inference mode.

        Returns
        -------
        float
            Learned bias of the model, ``0.0`` for a model without one.
        """

    @abstractmethod
    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """
        Run the image tower on RGB images and pool its tokens as ``image_pooling`` says.

        Called in inference mode.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.

        Returns
        -------
        torch.Tensor
            Unnormalized image embeddings in the joint space, shape (B, D).
        """

    @abstractmethod
    def _encode_sentences(self, sentences: Sequence[str]) -> torch.Tensor:
        """
        Run the text tower on filled templates.

        Called in inference mode with at most ``TEXT_BATCH_SIZE`` sentences.

        Parameters
        ----------
        sentences : Sequence[str]
            Sentences to embed.

        Returns
        -------
        torch.Tensor
            Unnormalized text embeddings in the joint space, shape (S, D).
        """

    @property
    def logit_scale(self) -> float:
        """
        Factor turning cosine similarities into logits, read from the model once after loading.

        Returns
        -------
        float
            Exponentiated learned temperature of the model.
        """
        return self._logit_scale

    @property
    def logit_bias(self) -> float:
        """
        Offset added to the scaled cosine similarities, read from the model once after loading.

        Returns
        -------
        float
            Learned bias of the model, ``0.0`` for a model without one.
        """
        return self._logit_bias

    def _embed_mini_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        return self._to_array(self._embed_image_tensor(images))

    def _embed_texts(self, texts: Sequence[str]) -> FloatArray:
        return self._to_array(self._query_embeddings.embed_texts(texts))

    def _embed_prompt(self, prompt: Prompt) -> FloatArray:
        return self._to_array(self._query_embeddings.embed(prompt))

    def _embed_image_tensor(self, images: Sequence[Image.Image]) -> torch.Tensor:
        with torch.inference_mode():
            image_embeddings: torch.Tensor = self._encode_images(images)
        return torch.nn.functional.normalize(image_embeddings.float(), dim=-1)

    def _embed_sentences(self, sentences: Sequence[str]) -> torch.Tensor:
        sentence_embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(sentences), self.TEXT_BATCH_SIZE):
                chunk: Sequence[str] = sentences[start : start + self.TEXT_BATCH_SIZE]
                sentence_embeddings.append(self._encode_sentences(chunk).float())
        return torch.cat(sentence_embeddings, dim=0)

    def _embed_references(self, references: Sequence[VisualReference]) -> torch.Tensor:
        images: list[Image.Image] = [self._to_rgb(reference.image) for reference in references]
        mini_batches: list[tuple[Image.Image, ...]] = list(batched(images, self.batch_size))
        return torch.cat([self._embed_image_tensor(mini_batch) for mini_batch in mini_batches], dim=0)

    def _tokenize(self, sentences: Sequence[str]) -> BatchEncoding:
        """
        Tokenize sentences padded to the text positions of the model, as the text towers were trained.

        Parameters
        ----------
        sentences : Sequence[str]
            Sentences to tokenize.

        Returns
        -------
        BatchEncoding
            ``input_ids`` and ``attention_mask`` on the runtime device, shape (S, ``_text_token_limit``).

        Raises
        ------
        ValueError
            If a sentence does not fit in ``_text_token_limit`` tokens.
        """
        encoding: BatchEncoding = self._processor.tokenizer(
            list(sentences), padding="max_length", max_length=self._text_token_limit
        )
        token_ids: list[list[int]] = encoding["input_ids"]
        longest_sentence: int = max(range(len(token_ids)), key=lambda index: len(token_ids[index]))
        token_count: int = len(token_ids[longest_sentence])
        if token_count > self._text_token_limit:
            raise ValueError(
                f"text queries filled into the templates must fit in {self._text_token_limit} tokens. "
                + f"got {token_count} tokens for {sentences[longest_sentence]!r}"
            )
        encoding.convert_to_tensors("pt")
        return encoding.to(self._runtime.device)

    @staticmethod
    def _to_array(tensor: torch.Tensor) -> FloatArray:
        return tensor.float().cpu().numpy().astype(np.float64)
