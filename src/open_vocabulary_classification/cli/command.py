import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, TextIO

from ..classifier import OpenVocabularyClassifier
from ..image_tiling import ImageTiling
from ..options import ClassifierBackend, Device, ImagePooling, Precision
from ..prompt import Prompt, PromptQuery, TextQuery, VisualQuery, VisualReference
from ..result import Classification, ClassificationRecord
from ..settings import ClassifierSettings, PresetCatalog
from .image_file_stream import ImageFileStream
from .image_files import ImageFiles
from .output_format import OutputFormat


class ClassifyCommand:
    """
    Command line classifying image files and directories into the classes given as arguments.

    Settings come from a packaged preset (``--preset clip/openai_vit_b16``), a YAML file in the preset format
    (``--config``) or a backend and weights (``--backend``, ``--weights``); ``--pooling``, ``--templates``,
    ``--batch-size``, ``--device`` and ``--precision`` override them. A class written as ``name:query,query`` (e.g.
    ``dog:dog,puppy``) is queried by each query and reported as ``name``; a query ``@path`` is a visual query whose
    references are the image file or the images of the directory at ``path`` (e.g. ``mug:@refs/mug``). A class given
    more than once collects the queries of every argument. Images are read one mini-batch at a time, upright as their
    EXIF orientation says; files that cannot be decoded are reported on the error stream and skipped. Invalid
    arguments are reported before the model is loaded.

    Installed as the ``open-vocabulary-classify`` script.
    """

    CLASS_QUERY_SEPARATOR: ClassVar[str] = ":"
    QUERY_SEPARATOR: ClassVar[str] = ","
    VISUAL_QUERY_PREFIX: ClassVar[str] = "@"
    PRESET_SEPARATOR: ClassVar[str] = "/"
    DEFAULT_TEMPLATES: ClassVar[tuple[str, ...]] = ("a photo of a {}.",)

    @classmethod
    def main(cls, arguments: Sequence[str] | None = None) -> None:
        """
        Run the command, printing results to standard output and warnings to standard error.

        Parameters
        ----------
        arguments : Sequence[str] | None
            Command-line arguments without the program name; ``None`` reads ``sys.argv``.
        """
        cls.run(sys.argv[1:] if arguments is None else arguments, sys.stdout, sys.stderr)

    @classmethod
    def run(cls, arguments: Sequence[str], output: TextIO, warnings: TextIO) -> None:
        """
        Parse the arguments, classify every image and write the classes of each one.

        Parameters
        ----------
        arguments : Sequence[str]
            Command-line arguments without the program name.
        output : TextIO
            Stream the results are written to.
        warnings : TextIO
            Stream skipped image files are reported on.

        Raises
        ------
        SystemExit
            If the arguments are invalid, e.g. a missing path, an unknown preset or invalid settings.
        """
        parser: argparse.ArgumentParser = cls._parser()
        namespace: argparse.Namespace = parser.parse_args(arguments)
        if namespace.list_presets:
            cls._write_presets(output)
            return
        cls._validate_arguments(parser, namespace)
        try:
            image_paths: list[Path] = ImageFiles.find([Path(path) for path in namespace.images], namespace.recursive)
            prompt: Prompt = cls.parse_prompt(namespace.classes)
            tiling: ImageTiling | None = (
                None if namespace.tile_grid is None else ImageTiling(namespace.tile_grid, namespace.tile_overlap)
            )
            settings: ClassifierSettings = cls._settings(parser, namespace)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(str(error.args[0]) if error.args else repr(error))
        classifier: OpenVocabularyClassifier = settings.build()
        output_format: OutputFormat = namespace.format
        images: ImageFileStream = ImageFileStream(image_paths, warnings)
        for result in classifier.iter_classify_images(images, prompt, tiling):
            classifications: tuple[Classification, ...] = (
                result.top_k(namespace.top_k)
                if namespace.threshold is None
                else result.filter_by_score(namespace.threshold)[: namespace.top_k]
            )
            output.write(cls._format(images.pending_paths.popleft(), classifications, output_format))
            output.flush()
        if images.skipped_paths:
            warnings.write(f"skipped {len(images.skipped_paths)} of {len(image_paths)} image files.\n")

    @classmethod
    def parse_prompt(cls, class_arguments: Sequence[str]) -> Prompt:
        """
        Build a prompt from ``name`` or ``name:query,query`` arguments.

        Parameters
        ----------
        class_arguments : Sequence[str]
            Command-line class arguments; a query ``@path`` names reference images.

        Returns
        -------
        Prompt
            Classes named ``name``, each queried by its listed queries or by its name.

        Raises
        ------
        FileNotFoundError
            If a reference path does not exist or holds no image.
        OSError
            If a reference image cannot be read.
        ValueError
            If a class name or query is blank or repeated.
        """
        class_queries: dict[str, list[PromptQuery]] = {}
        for argument in class_arguments:
            class_name, _, query_list = argument.partition(cls.CLASS_QUERY_SEPARATOR)
            stripped_name: str = class_name.strip()
            query_arguments: list[str] = query_list.split(cls.QUERY_SEPARATOR) if query_list else [stripped_name]
            class_queries.setdefault(stripped_name, []).extend(cls._parse_query(query) for query in query_arguments)
        return Prompt(class_queries)

    @classmethod
    def _parse_query(cls, query_argument: str) -> PromptQuery:
        stripped_query: str = query_argument.strip()
        if not stripped_query.startswith(cls.VISUAL_QUERY_PREFIX):
            return TextQuery(stripped_query)
        reference_paths: list[Path] = ImageFiles.find(
            [Path(stripped_query.removeprefix(cls.VISUAL_QUERY_PREFIX))], is_recursive=False
        )
        return VisualQuery(tuple(VisualReference(ImageFiles.load(path)) for path in reference_paths))

    @staticmethod
    def _validate_arguments(parser: argparse.ArgumentParser, namespace: argparse.Namespace) -> None:
        if not namespace.images or not namespace.classes:
            parser.error("images and --classes are required.")
        if namespace.top_k <= 0:
            parser.error(f"--top-k must be positive. got {namespace.top_k}")
        if namespace.threshold is not None and not 0.0 <= namespace.threshold <= 1.0:
            parser.error(f"--threshold must be in [0, 1]. got {namespace.threshold}")
        if namespace.backend is not None and namespace.weights is None:
            parser.error("--backend only applies to --weights; a preset or config file names its own backend.")

    @classmethod
    def _parser(cls) -> argparse.ArgumentParser:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            prog="open-vocabulary-classify",
            description="Classify images into classes given as text or reference images.",
        )
        parser.add_argument("images", nargs="*", help="image files or directories")
        parser.add_argument(
            "--classes", nargs="+", help="class names, or name:query,query; a query @path uses reference images"
        )
        parser.add_argument("--recursive", action="store_true", help="search sub-directories too")
        parser.add_argument("--preset", help="packaged preset as backend/name, e.g. clip/openai_vit_b16")
        parser.add_argument("--config", type=Path, help="YAML file in the preset format")
        parser.add_argument("--weights", help="Hugging Face Hub model id or local checkpoint; needs --backend")
        parser.add_argument("--backend", type=ClassifierBackend, choices=list(ClassifierBackend))
        parser.add_argument("--pooling", type=ImagePooling, choices=list(ImagePooling))
        parser.add_argument("--templates", nargs="+", help="text templates with one {} each")
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--device", type=Device, choices=list(Device))
        parser.add_argument("--precision", type=Precision, choices=list(Precision))
        parser.add_argument("--tile-grid", type=int, help="also classify a grid of this many tiles per side")
        parser.add_argument("--tile-overlap", type=float, default=0.25, help="overlap ratio of neighboring tiles")
        parser.add_argument("--top-k", type=int, default=3, help="maximum number of classes printed per image")
        parser.add_argument("--threshold", type=float, help="print only classes scoring at least this")
        parser.add_argument("--format", type=OutputFormat, choices=list(OutputFormat), default=OutputFormat.TEXT)
        parser.add_argument("--list-presets", action="store_true", help="print the packaged presets and exit")
        return parser

    @classmethod
    def _settings(cls, parser: argparse.ArgumentParser, namespace: argparse.Namespace) -> ClassifierSettings:
        sources: list[object] = [namespace.preset, namespace.config, namespace.weights]
        if sum(source is not None for source in sources) != 1:
            parser.error("exactly one of --preset, --config or --weights is required.")
        base: ClassifierSettings
        if namespace.preset is not None:
            backend_name, _, preset_name = str(namespace.preset).partition(cls.PRESET_SEPARATOR)
            backend_names: list[str] = [backend.value for backend in ClassifierBackend]
            if backend_name not in backend_names:
                parser.error(f"--preset must be backend/name with a backend of {backend_names}. got {namespace.preset}")
            base = PresetCatalog.load(ClassifierBackend(backend_name), preset_name)
        elif namespace.config is not None:
            base = PresetCatalog.load_file(namespace.config)
        else:
            if namespace.backend is None:
                parser.error("--weights needs --backend.")
            backend: ClassifierBackend = namespace.backend
            base = ClassifierSettings(
                backend=backend,
                weights_path=namespace.weights,
                image_pooling=backend.trained_image_pooling,
                text_templates=cls.DEFAULT_TEMPLATES,
            )
        return replace(
            base,
            image_pooling=base.image_pooling if namespace.pooling is None else namespace.pooling,
            text_templates=base.text_templates if namespace.templates is None else tuple(namespace.templates),
            batch_size=base.batch_size if namespace.batch_size is None else namespace.batch_size,
            device=base.device if namespace.device is None else namespace.device,
            precision=base.precision if namespace.precision is None else namespace.precision,
        )

    @staticmethod
    def _write_presets(output: TextIO) -> None:
        for backend in ClassifierBackend:
            for name in PresetCatalog.names(backend):
                output.write(f"{backend.value}/{name}\n")

    @staticmethod
    def _format(path: Path, classifications: Sequence[Classification], output_format: OutputFormat) -> str:
        records: list[ClassificationRecord] = [classification.to_record() for classification in classifications]
        match output_format:
            case OutputFormat.JSON_LINES:
                return json.dumps({"image": str(path), "classifications": records}) + "\n"
            case OutputFormat.TEXT:
                lines: list[str] = [f"{path}:"]
                for record in records:
                    matched_query: str = (
                        f"<{record['matched_query_reference_count']} reference images>"
                        if record["matched_query_text"] is None
                        else record["matched_query_text"]
                    )
                    lines.append(
                        f"  {record['class_name']:20s} {matched_query:20s} "
                        + f"score {record['score']:.3f}  logit {record['logit']:.2f}"
                    )
                return "\n".join(lines) + "\n"
