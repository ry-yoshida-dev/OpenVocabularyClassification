from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .array_types import FloatArray
from .image_tiling import ImageTiling
from .options import ClassifierBackend, ImagePooling, ScoreActivation
from .prompt import Prompt, TextQuery
from .result import ClassificationResult
from .settings import ClassifierSettings


class OpenVocabularyClassifier(ABC):
    """
    Common base of every open-vocabulary image classifier, independent of the inference runtime.

    Images and queries are embedded into one joint space; the logit of an image and a query is the cosine similarity
    of their L2-normalized embeddings times ``logit_scale`` plus ``logit_bias``. Each class takes the logit of its best
    query and the activation of ``BACKEND`` scores the classes. Subclasses embed images, text queries and prompts and
    expose the scale and bias of their model; how the model runs (PyTorch, ONNX Runtime, TensorRT, ...) is up to each
    subclass and its runtime. Everything else, mini-batching, RGB conversion, tiling, scoring and shape checks, is
    shared here.

    Images of any mode are converted to RGB; transparent pixels are composited over ``TRANSPARENCY_BACKGROUND``
    (white) instead of keeping the color hidden under them, and single-channel images of more than 8 bits (16-bit,
    32-bit integer or float) are stretched linearly from their lowest to their highest finite value instead of being
    clipped to 255.

    Image embeddings are pooled as ``settings.image_pooling`` says, so ``embed_images`` returns exactly the
    embeddings ``classify_images`` scores: ``classify_embeddings(embed_images(images, tiling), prompt)`` equals
    ``classify_images(images, prompt, tiling)`` and lets many prompts reuse one pass of the image tower. With a tiling
    every image has one embedding per view, shape (V, D), and each query keeps its highest logit over the views.

    Attributes
    ----------
    settings : ClassifierSettings
        Backend, weights, pooling, text templates, batching, device and precision.
    """

    BACKEND: ClassVar[ClassifierBackend]
    TRANSPARENCY_BACKGROUND: ClassVar[tuple[int, int, int]] = (255, 255, 255)
    UNIT_NORM_TOLERANCE: ClassVar[float] = 1e-3
    HIGH_BIT_DEPTH_MODES: ClassVar[frozenset[str]] = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N", "F"})
    MAX_CHANNEL_VALUE: ClassVar[float] = 255.0

    def __init__(self, settings: ClassifierSettings) -> None:
        """
        Parameters
        ----------
        settings : ClassifierSettings
            Backend, weights, pooling, text templates, batching, device and precision.

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
            Mini-batch size of the image tower.
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

    @property
    @abstractmethod
    def logit_scale(self) -> float:
        """
        Factor turning cosine similarities into logits.

        Returns
        -------
        float
            Exponentiated learned temperature of the model.
        """

    @property
    @abstractmethod
    def logit_bias(self) -> float:
        """
        Offset added to the scaled cosine similarities.

        Returns
        -------
        float
            Learned bias of the model, ``0.0`` for a model without one.
        """

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
    def _embed_texts(self, texts: Sequence[str]) -> FloatArray:
        """
        Embed stripped, non-blank text queries through the text templates.

        Parameters
        ----------
        texts : Sequence[str]
            Text queries.

        Returns
        -------
        FloatArray
            L2-normalized text embeddings in input order, shape (T, D).
        """

    @abstractmethod
    def _embed_prompt(self, prompt: Prompt) -> FloatArray:
        """
        Embed every query of a prompt.

        Parameters
        ----------
        prompt : Prompt
            Prompt with text queries, visual queries or both.

        Returns
        -------
        FloatArray
            L2-normalized query embeddings in query-id order, shape (Q, D).
        """

    def classify(self, image: Image.Image, prompt: Prompt, tiling: ImageTiling | None = None) -> ClassificationResult:
        """
        Score the prompt classes for a single image.

        Parameters
        ----------
        image : Image.Image
            Input image.
        prompt : Prompt
            Classes to choose between.
        tiling : ImageTiling | None
            Tiles to classify on top of or instead of the whole image; ``None`` classifies the whole image only.

        Returns
        -------
        ClassificationResult
            Class scores of the image.
        """
        return self.classify_images([image], prompt, tiling)[0]

    def classify_images(
        self, images: Sequence[Image.Image], prompt: Prompt, tiling: ImageTiling | None = None
    ) -> list[ClassificationResult]:
        """
        Score the prompt classes for many images, processed in mini-batches.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Input images of arbitrary size and mode.
        prompt : Prompt
            Classes to choose between.
        tiling : ImageTiling | None
            Tiles to classify on top of or instead of each whole image; every query keeps its highest logit over the
            views. ``None`` classifies the whole images only.

        Returns
        -------
        list[ClassificationResult]
            One result per image, in input order.

        Raises
        ------
        ValueError
            If ``images`` is empty.
        RuntimeError
            If the model does not return one embedding per image or query.
        """
        self._validate_images(images)
        return list(self.iter_classify_images(images, prompt, tiling))

    def iter_classify_images(
        self, images: Iterable[Image.Image], prompt: Prompt, tiling: ImageTiling | None = None
    ) -> Iterator[ClassificationResult]:
        """
        Score the prompt classes for a stream of images, reading one mini-batch at a time.

        Only one mini-batch of images is held at once, so a generator loading images lazily classifies datasets of
        any size in bounded memory. With a tiling, the views of several images fill one mini-batch together. The
        prompt is embedded when the first result is requested.

        Parameters
        ----------
        images : Iterable[Image.Image]
            Input images of arbitrary size and mode; may be a generator.
        prompt : Prompt
            Classes to choose between.
        tiling : ImageTiling | None
            Tiles to classify on top of or instead of each whole image; ``None`` classifies the whole images only.

        Yields
        ------
        ClassificationResult
            Result of each image, in input order; nothing for an empty stream.

        Raises
        ------
        RuntimeError
            If the model does not return one embedding per image or query.
        """
        query_embeddings: FloatArray = self.embed_prompt(prompt)
        for image_embeddings in self._iter_embedding_batches(images, tiling):
            for query_logits in self._logits(image_embeddings, query_embeddings):
                yield self._result(query_logits, prompt)

    def classify_embeddings(self, image_embeddings: FloatArray, prompt: Prompt) -> list[ClassificationResult]:
        """
        Score the prompt classes for images embedded beforehand by ``embed_images``, without running the image tower.

        Embedding a dataset once and classifying it with many prompts, thresholds or class sets costs one pass of the
        image tower in total. The embeddings must come from a classifier with the same weights and pooling. View
        embeddings of a tiling, shape (N, V, D), keep the highest logit of each query over the views of an image.

        Parameters
        ----------
        image_embeddings : FloatArray
            L2-normalized image embeddings, shape (N, D), or view embeddings, shape (N, V, D).
        prompt : Prompt
            Classes to choose between.

        Returns
        -------
        list[ClassificationResult]
            One result per image, in input order.

        Raises
        ------
        ValueError
            If the embeddings are not a non-empty (N, D) or (N, V, D) array of finite unit vectors of the query
            dimension.
        """
        if image_embeddings.ndim not in (2, 3) or 0 in image_embeddings.shape[:-1]:
            raise ValueError(
                f"image_embeddings must have shape (N, D) or (N, V, D) with N, V > 0. got {image_embeddings.shape}"
            )
        if not bool(np.isfinite(image_embeddings).all()):
            raise ValueError("image_embeddings must be finite.")
        if not np.allclose(np.linalg.norm(image_embeddings, axis=-1), 1.0, atol=self.UNIT_NORM_TOLERANCE):
            raise ValueError("image_embeddings must be L2-normalized, as embed_images returns them.")
        query_embeddings: FloatArray = self.embed_prompt(prompt)
        if image_embeddings.shape[-1] != query_embeddings.shape[1]:
            raise ValueError(
                f"image_embeddings must have dimension {query_embeddings.shape[1]}. got {image_embeddings.shape[-1]}"
            )
        return [self._result(query_logits, prompt) for query_logits in self._logits(image_embeddings, query_embeddings)]

    def embed_image(self, image: Image.Image, tiling: ImageTiling | None = None) -> FloatArray:
        """
        Embed a single image into the joint image-text space.

        Parameters
        ----------
        image : Image.Image
            Input image.
        tiling : ImageTiling | None
            Views to embed; ``None`` embeds the whole image only.

        Returns
        -------
        FloatArray
            L2-normalized embedding, shape (D,), or one per view, shape (V, D), with a tiling.
        """
        image_embedding: FloatArray = self.embed_images([image], tiling)[0]
        return image_embedding

    def embed_images(self, images: Sequence[Image.Image], tiling: ImageTiling | None = None) -> FloatArray:
        """
        Embed many images into the joint image-text space, processed in mini-batches.

        The embeddings are those scored against the prompt queries, pooled as ``image_pooling`` says, so dot products
        compare images with each other (``embed_images``) or with text (``embed_texts``) in the space the classifier
        uses, and ``classify_embeddings`` classifies them later.

        Parameters
        ----------
        images : Sequence[Image.Image]
            Input images of arbitrary size and mode.
        tiling : ImageTiling | None
            Views to embed of every image; ``None`` embeds the whole images only.

        Returns
        -------
        FloatArray
            L2-normalized embeddings in input order, shape (N, D), or (N, V, D) with the views of ``tiling`` in the
            order of ``ImageTiling.views``.

        Raises
        ------
        ValueError
            If ``images`` is empty.
        RuntimeError
            If the model does not return one embedding per image.
        """
        self._validate_images(images)
        image_embeddings: FloatArray = np.concatenate(list(self._iter_embedding_batches(images, tiling)))
        return image_embeddings

    def iter_embed_images(
        self, images: Iterable[Image.Image], tiling: ImageTiling | None = None
    ) -> Iterator[FloatArray]:
        """
        Embed a stream of images, reading one mini-batch at a time.

        Parameters
        ----------
        images : Iterable[Image.Image]
            Input images of arbitrary size and mode; may be a generator.
        tiling : ImageTiling | None
            Views to embed of every image; ``None`` embeds the whole images only.

        Yields
        ------
        FloatArray
            L2-normalized embedding of each image, shape (D,), or (V, D) with a tiling, in input order; nothing for
            an empty stream.

        Raises
        ------
        RuntimeError
            If the model does not return one embedding per image.
        """
        for image_embeddings in self._iter_embedding_batches(images, tiling):
            yield from image_embeddings

    def embed_texts(self, texts: Sequence[str]) -> FloatArray:
        """
        Embed text queries into the joint image-text space, e.g. to search images by text.

        Every text is filled into the text templates and ensembled exactly as a ``TextQuery`` of a prompt, so
        ``embed_images(images) @ embed_texts(texts).T`` are the cosine similarities a classifier scores. Embeddings are
        cached per text.

        Parameters
        ----------
        texts : Sequence[str]
            Text queries, e.g. ``("dog", "a red car")``; surrounding whitespace is stripped.

        Returns
        -------
        FloatArray
            L2-normalized embeddings in input order, shape (T, D).

        Raises
        ------
        ValueError
            If ``texts`` is empty or a text is blank.
        RuntimeError
            If the model does not return one embedding per text.
        """
        if not texts:
            raise ValueError("texts must contain at least one text.")
        stripped_texts: list[str] = [TextQuery(text).text for text in texts]
        return self._checked_embeddings(self._embed_texts(stripped_texts), len(stripped_texts))

    def embed_prompt(self, prompt: Prompt) -> FloatArray:
        """
        Embed every query of a prompt, text queries through the templates and visual queries as reference averages.

        Parameters
        ----------
        prompt : Prompt
            Prompt with text queries, visual queries or both.

        Returns
        -------
        FloatArray
            L2-normalized query embeddings in query-id order, shape (Q, D).

        Raises
        ------
        RuntimeError
            If the model does not return one embedding per query.
        """
        return self._checked_embeddings(self._embed_prompt(prompt), len(prompt.queries))

    def _iter_embedding_batches(
        self, images: Iterable[Image.Image], tiling: ImageTiling | None
    ) -> Iterator[FloatArray]:
        match tiling:
            case None:
                for mini_batch in batched(images, self.batch_size):
                    yield self._embed_checked(mini_batch)
            case ImageTiling():
                images_per_batch: int = max(1, self.batch_size // tiling.view_count)
                for image_batch in batched(images, images_per_batch):
                    views: list[Image.Image] = [
                        view for image in image_batch for view in tiling.views(self._to_rgb(image))
                    ]
                    view_embeddings: FloatArray = np.concatenate(
                        [self._embed_checked(mini_batch) for mini_batch in batched(views, self.batch_size)]
                    )
                    yield view_embeddings.reshape(len(image_batch), tiling.view_count, -1)

    def _embed_checked(self, images: Sequence[Image.Image]) -> FloatArray:
        return self._checked_embeddings(self._embed_mini_batch([self._to_rgb(image) for image in images]), len(images))

    def _logits(self, image_embeddings: FloatArray, query_embeddings: FloatArray) -> FloatArray:
        logits: FloatArray = image_embeddings @ query_embeddings.T * self.logit_scale + self.logit_bias
        return logits if logits.ndim == 2 else logits.max(axis=1)

    def _result(self, query_logits: FloatArray, prompt: Prompt) -> ClassificationResult:
        return ClassificationResult(prompt=prompt, query_logits=query_logits, score_activation=self.score_activation)

    @staticmethod
    def _checked_embeddings(embeddings: FloatArray, expected_count: int) -> FloatArray:
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_count:
            raise RuntimeError(f"expected embeddings of shape ({expected_count}, D). got {embeddings.shape}")
        return embeddings

    @staticmethod
    def _validate_images(images: Sequence[Image.Image]) -> None:
        if not images:
            raise ValueError("images must contain at least one image.")

    @classmethod
    def _to_rgb(cls, image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image
        if image.mode in cls.HIGH_BIT_DEPTH_MODES:
            return cls._stretched_to_rgb(image)
        if not image.has_transparency_data:
            return image.convert("RGB")
        rgba_image: Image.Image = image.convert("RGBA")
        background: Image.Image = Image.new("RGBA", rgba_image.size, cls.TRANSPARENCY_BACKGROUND)
        return Image.alpha_composite(background, rgba_image).convert("RGB")

    @classmethod
    def _stretched_to_rgb(cls, image: Image.Image) -> Image.Image:
        values: FloatArray = np.asarray(image, dtype=np.float64)
        finite_values: FloatArray = values[np.isfinite(values)]
        lowest: float = float(finite_values.min()) if finite_values.size else 0.0
        value_range: float = float(finite_values.max()) - lowest if finite_values.size else 0.0
        scale: float = cls.MAX_CHANNEL_VALUE / value_range if value_range > 0.0 else 0.0
        stretched: FloatArray = np.nan_to_num(
            (values - lowest) * scale, nan=0.0, posinf=cls.MAX_CHANNEL_VALUE, neginf=0.0
        )
        channel: NDArray[np.uint8] = np.clip(stretched, 0.0, cls.MAX_CHANNEL_VALUE).round().astype(np.uint8)
        return Image.fromarray(channel).convert("RGB")
