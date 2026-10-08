from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
from PIL import Image

from .array_types import FloatArray
from .options import ClassifierBackend, ImagePooling, ScoreActivation
from .prompt import Prompt
from .result import ClassificationResult
from .settings import ClassifierSettings


class OpenVocabularyClassifier(ABC):
    """
    Common base of every open-vocabulary image classifier, independent of the inference runtime.

    Subclasses declare the ``BACKEND`` they implement, which fixes the image poolings they offer and how their logits
    become scores. The base owns the settings, mini-batching, RGB conversion and the post-processing shared by every
    backend: each class takes the logit of its best query and the activation of the backend scores the classes. How
    the model runs (PyTorch, ONNX Runtime, TensorRT, ...) is up to each subclass and its runtime.

    Images of any mode are converted to RGB; transparent pixels are composited over ``TRANSPARENCY_BACKGROUND``
    (white) instead of keeping the color hidden under them.

    Image embeddings are pooled as ``settings.image_pooling`` says, both for classification and for
    ``embed_images``, so the same embedding can be used for retrieval or clustering outside a prompt.

    Attributes
    ----------
    settings : ClassifierSettings
        Backend, weights, pooling, text templates, batching and device.
    """

    BACKEND: ClassVar[ClassifierBackend]
    TRANSPARENCY_BACKGROUND: ClassVar[tuple[int, int, int]] = (255, 255, 255)

    def __init__(self, settings: ClassifierSettings) -> None:
        """
        Parameters
        ----------
        settings : ClassifierSettings
            Backend, weights, pooling, text templates, batching and device.

        Raises
        ------
        ValueError
            If the settings are for another backend.
        """
        if settings.backend is not self.BACKEND:
            raise ValueError(f"{type(self).__name__} needs {self.BACKEND} settings. got {settings.backend}")
        self.settings: ClassifierSettings = settings

    @property
    def batch_size(self) -> int:
        """
        Number of images processed per forward pass.

        Returns
        -------
        int
            Mini-batch size used by ``classify_images`` and ``embed_images``.
        """
        return self.settings.batch_size

    @property
    def image_pooling(self) -> ImagePooling:
        """
        Pooling producing the image embeddings.

        Returns
        -------
        ImagePooling
            Configured pooling.
        """
        return self.settings.image_pooling

    @property
    def score_activation(self) -> ScoreActivation:
        """
        Activation turning class logits into scores.

        Returns
        -------
        ScoreActivation
            Activation of ``BACKEND``.
        """
        return self.BACKEND.score_activation

    @abstractmethod
    def _embed_mini_batch(self, images: Sequence[Image.Image]) -> FloatArray:
        """
        Embed at most ``batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.

        Returns
        -------
        FloatArray
            L2-normalized image embeddings in input order, shape (B, D).
        """

    @abstractmethod
    def _score_mini_batch(self, images: Sequence[Image.Image], prompt: Prompt) -> FloatArray:
        """
        Score every query of a prompt against at most ``batch_size`` RGB images.

        Parameters
        ----------
        images : Sequence[Image.Image]
            RGB images.
        prompt : Prompt
            Classes to choose between.

        Returns
        -------
        FloatArray
            Logit of every query for every image, shape (B, Q).
        """

    def classify(self, image: Image.Image, prompt: Prompt) -> ClassificationResult:
        """
        Score the prompt classes for a single image.

        Parameters
        ----------
        image : Image.Image
            Input image.
        prompt : Prompt
            Classes to choose between.

        Returns
        -------
        ClassificationResult
            Class scores of the image.
        """
        return self.classify_images([image], prompt)[0]

    def classify_images(self, images: Sequence[Image.Image], prompt: Prompt) -> list[ClassificationResult]:
        """
        Score the prompt classes for many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Input images of arbitrary size and mode.
        prompt : Prompt
            Classes to choose between.

        Returns
        -------
        list[ClassificationResult]
            One result per image, in input order.

        Raises
        ------
        ValueError
            If ``images`` is empty.
        RuntimeError
            If the model does not return one logit per image and query.
        """
        self._validate_images(images)
        results: list[ClassificationResult] = []
        for mini_batch in self._mini_batches(images):
            query_logits: FloatArray = self._score_mini_batch([self._to_rgb(image) for image in mini_batch], prompt)
            expected_shape: tuple[int, int] = (len(mini_batch), len(prompt.queries))
            if query_logits.shape != expected_shape:
                raise RuntimeError(f"expected query logits of shape {expected_shape}. got {query_logits.shape}")
            results.extend(
                ClassificationResult(prompt=prompt, query_logits=logits, score_activation=self.score_activation)
                for logits in query_logits
            )
        return results

    def embed_image(self, image: Image.Image) -> FloatArray:
        """
        Embed a single image into the joint image-text space.

        Parameters
        ----------
        image : Image.Image
            Input image.

        Returns
        -------
        FloatArray
            L2-normalized embedding, shape (D,).
        """
        image_embedding: FloatArray = self.embed_images([image])[0]
        return image_embedding

    def embed_images(self, images: Sequence[Image.Image]) -> FloatArray:
        """
        Embed many images into the joint image-text space, processed in mini-batches.

        The embeddings are those scored against the prompt queries, pooled as ``image_pooling`` says, so cosine
        similarities between them (dot products) compare images in the space the classifier uses.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Input images of arbitrary size and mode.

        Returns
        -------
        FloatArray
            L2-normalized embeddings in input order, shape (N, D).

        Raises
        ------
        ValueError
            If ``images`` is empty.
        RuntimeError
            If the model does not return one embedding per image.
        """
        self._validate_images(images)
        embeddings: list[FloatArray] = []
        for mini_batch in self._mini_batches(images):
            mini_batch_embeddings: FloatArray = self._embed_mini_batch([self._to_rgb(image) for image in mini_batch])
            if mini_batch_embeddings.ndim != 2 or mini_batch_embeddings.shape[0] != len(mini_batch):
                raise RuntimeError(
                    f"expected embeddings of shape ({len(mini_batch)}, D). got {mini_batch_embeddings.shape}"
                )
            embeddings.append(mini_batch_embeddings)
        image_embeddings: FloatArray = np.concatenate(embeddings, axis=0)
        return image_embeddings

    @staticmethod
    def _validate_images(images: Sequence[Image.Image]) -> None:
        if not images:
            raise ValueError("images must contain at least one image.")

    def _mini_batches(self, images: Sequence[Image.Image]) -> Iterator[Sequence[Image.Image]]:
        for start in range(0, len(images), self.batch_size):
            yield images[start : start + self.batch_size]

    @classmethod
    def _to_rgb(cls, image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image
        if not image.has_transparency_data:
            return image.convert("RGB")
        rgba_image: Image.Image = image.convert("RGBA")
        background: Image.Image = Image.new("RGBA", rgba_image.size, cls.TRANSPARENCY_BACKGROUND)
        return Image.alpha_composite(background, rgba_image).convert("RGB")
