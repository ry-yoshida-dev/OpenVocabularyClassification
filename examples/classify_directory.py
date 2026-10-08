"""
Classify every image of a directory into a fixed set of classes.

Usage
-----
python examples/classify_directory.py IMAGE_DIR --backend clip --weights openai/clip-vit-base-patch32 --classes cat dog

A class written as ``name:query,query`` (e.g. ``dog:dog,puppy``) is queried by each phrase and reported as ``name``.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from open_vocabulary_classification import (
    ClassificationResult,
    ClassifierBackend,
    ClassifierSettings,
    Device,
    ImagePooling,
    OpenVocabularyClassifier,
    Prompt,
    PromptQuery,
    TextQuery,
    VisualQuery,
)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
CLASS_QUERY_SEPARATOR: str = ":"
QUERY_SEPARATOR: str = ","


def parse_prompt(class_arguments: list[str]) -> Prompt:
    """
    Build a text prompt from ``name`` or ``name:query,query`` arguments.

    A class given more than once collects the phrases of every argument, e.g. ``dog:dog,puppy dog:hound``.

    Parameters
    ----------
    class_arguments : list[str]
        Command-line class arguments.

    Returns
    -------
    Prompt
        Classes named ``name``, each queried by its listed phrases or by its name.
    """
    class_texts: dict[str, list[str]] = {}
    for argument in class_arguments:
        class_name, _, phrases = argument.partition(CLASS_QUERY_SEPARATOR)
        stripped_name: str = class_name.strip()
        class_phrases: list[str] = phrases.split(QUERY_SEPARATOR) if phrases else [stripped_name]
        class_texts.setdefault(stripped_name, []).extend(class_phrases)
    return Prompt.from_texts(class_texts)


def describe_query(query: PromptQuery) -> str:
    """
    Short label of the query that matched a class.

    Parameters
    ----------
    query : PromptQuery
        Matched query.

    Returns
    -------
    str
        The phrase of a text query, or the reference count of a visual query.
    """
    match query:
        case TextQuery(text=text):
            return text
        case VisualQuery(references=references):
            return f"<{len(references)} reference images>"


def load_image(path: Path) -> Image.Image:
    """
    Read an image into memory upright, closing its file.

    Parameters
    ----------
    path : Path
        Image file.

    Returns
    -------
    Image.Image
        Decoded image rotated as its EXIF orientation says.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image)


def main() -> None:
    """
    Parse arguments, run classification and print the top classes of each image.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--backend", type=ClassifierBackend, choices=list(ClassifierBackend), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--pooling", type=ImagePooling, choices=list(ImagePooling), default=None)
    parser.add_argument("--templates", nargs="+", default=["a photo of a {}."])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--half", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()

    image_directory: Path = arguments.image_directory
    image_paths: list[Path] = sorted(
        path for path in image_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_directory}")

    backend: ClassifierBackend = arguments.backend
    native_pooling: ImagePooling = (
        ImagePooling.CLASS_TOKEN if backend is ClassifierBackend.CLIP else ImagePooling.ATTENTION
    )
    classifier: OpenVocabularyClassifier = ClassifierSettings(
        backend=backend,
        weights_path=arguments.weights,
        image_pooling=native_pooling if arguments.pooling is None else arguments.pooling,
        text_templates=tuple(arguments.templates),
        batch_size=arguments.batch_size,
        device=arguments.device,
        is_half_precision_enabled=arguments.half,
    ).build()
    prompt: Prompt = parse_prompt(arguments.classes)

    for start in range(0, len(image_paths), classifier.batch_size):
        batch_paths: list[Path] = image_paths[start : start + classifier.batch_size]
        images: list[Image.Image] = [load_image(path) for path in batch_paths]
        results: list[ClassificationResult] = classifier.classify_images(images, prompt)
        for path, result in zip(batch_paths, results, strict=True):
            print(f"{path.name}:")
            for classification in result.top_k(arguments.top_k):
                print(
                    f"  {classification.class_name:20s} {describe_query(classification.matched_query):20s} "
                    + f"score {classification.score:.3f}  logit {classification.logit:.2f}"
                )


if __name__ == "__main__":
    main()
