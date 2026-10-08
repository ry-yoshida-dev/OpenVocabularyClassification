"""
Search a directory of images by text, and classify the same embeddings with several prompts.

Usage
-----
python examples/search_images.py IMAGE_DIR --preset siglip/v2_base_patch16_224 --query "a dog on a beach" --top-k 5

The image tower runs once per image: the embeddings are searched by text and then classified indoor/outdoor and by
time of day without embedding the images again.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from open_vocabulary_classification import (
    ClassificationResult,
    ClassifierBackend,
    OpenVocabularyClassifier,
    PresetCatalog,
    Prompt,
)
from open_vocabulary_classification.array_types import FloatArray, IntArray
from open_vocabulary_classification.cli import ImageFiles


@dataclass(frozen=True)
class ImageSearch:
    """
    Image embeddings of a directory, searchable by text and classifiable by any prompt.

    Attributes
    ----------
    classifier : OpenVocabularyClassifier
        Classifier that embedded the images.
    image_paths : tuple[Path, ...]
        Image files in embedding order.
    image_embeddings : FloatArray
        L2-normalized embedding of each image, shape (N, D).
    """

    classifier: OpenVocabularyClassifier
    image_paths: tuple[Path, ...]
    image_embeddings: FloatArray

    @classmethod
    def of_directory(cls, classifier: OpenVocabularyClassifier, directory: Path) -> "ImageSearch":
        """
        Embed every image of a directory, reading one mini-batch at a time.

        Parameters
        ----------
        classifier : OpenVocabularyClassifier
            Classifier embedding the images.
        directory : Path
            Directory of image files.

        Returns
        -------
        ImageSearch
            Searchable embeddings.
        """
        image_paths: list[Path] = ImageFiles.find([directory], is_recursive=False)
        image_embeddings: FloatArray = np.stack(
            list(classifier.iter_embed_images(ImageFiles.load(path) for path in image_paths))
        )
        return cls(classifier=classifier, image_paths=tuple(image_paths), image_embeddings=image_embeddings)

    def search(self, query: str, top_k: int) -> dict[Path, float]:
        """
        Images most similar to a text.

        Parameters
        ----------
        query : str
            Text describing the wanted images.
        top_k : int
            Number of images returned.

        Returns
        -------
        dict[Path, float]
            Cosine similarity to the text of the ``top_k`` most similar image files, most similar first.
        """
        similarities: FloatArray = self.image_embeddings @ self.classifier.embed_texts([query])[0]
        ranked_ids: IntArray = np.argsort(-similarities, kind="stable")[:top_k].astype(np.int64)
        return {self.image_paths[image_id]: float(similarities[image_id]) for image_id in ranked_ids}

    def classify(self, prompt: Prompt) -> Sequence[ClassificationResult]:
        """
        Classify the embedded images without running the image tower again.

        Parameters
        ----------
        prompt : Prompt
            Classes to choose between.

        Returns
        -------
        Sequence[ClassificationResult]
            One result per image, in ``image_paths`` order.
        """
        return self.classifier.classify_embeddings(self.image_embeddings, prompt)


def main() -> None:
    """
    Parse arguments, embed the directory, then print the search hits and their classes under two prompts.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--preset", default="siglip/v2_base_patch16_224", help="packaged preset as backend/name")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    arguments: argparse.Namespace = parser.parse_args()

    backend_name, _, preset_name = str(arguments.preset).partition("/")
    classifier: OpenVocabularyClassifier = PresetCatalog.load(ClassifierBackend(backend_name), preset_name).build()
    image_search: ImageSearch = ImageSearch.of_directory(classifier, arguments.image_directory)
    prompts: tuple[Prompt, ...] = (
        Prompt.from_class_names(("indoor scene", "outdoor scene")),
        Prompt.from_class_names(("daytime", "night")),
    )
    results: list[Sequence[ClassificationResult]] = [image_search.classify(prompt) for prompt in prompts]
    image_ids: dict[Path, int] = {path: image_id for image_id, path in enumerate(image_search.image_paths)}

    for path, similarity in image_search.search(arguments.query, arguments.top_k).items():
        labels: list[str] = [prompt_results[image_ids[path]].top.class_name for prompt_results in results]
        print(f"{similarity:.3f}  {path.name}  ({', '.join(labels)})")


if __name__ == "__main__":
    main()
