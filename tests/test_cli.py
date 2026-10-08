import io
import json
from pathlib import Path

import pytest
from PIL import Image

from open_vocabulary_classification import (
    ClassifierBackend,
    ImagePooling,
    PresetCatalog,
    Prompt,
    TextQuery,
    VisualQuery,
)
from open_vocabulary_classification.cli import ClassifyCommand, ImageFiles


def run_with_warnings(arguments: list[str]) -> tuple[str, str]:
    output: io.StringIO = io.StringIO()
    warnings: io.StringIO = io.StringIO()
    ClassifyCommand.run(arguments, output, warnings)
    return output.getvalue(), warnings.getvalue()


def run(arguments: list[str]) -> str:
    output, _ = run_with_warnings(arguments)
    return output


def siglip_arguments(checkpoint_directory: Path) -> list[str]:
    return [
        "--weights",
        str(checkpoint_directory / "siglip"),
        "--backend",
        "siglip",
        "--device",
        "cpu",
        "--templates",
        "{}",
        "--format",
        "jsonl",
    ]


@pytest.fixture
def image_directory(tmp_path: Path) -> Path:
    directory: Path = tmp_path / "images"
    (directory / "nested").mkdir(parents=True)
    Image.new("RGB", (40, 30), (255, 0, 0)).save(directory / "b.png")
    Image.new("RGB", (30, 40), (0, 255, 0)).save(directory / "a.jpg")
    Image.new("RGB", (20, 20), (0, 0, 255)).save(directory / "nested" / "c.png")
    (directory / "notes.txt").write_text("not an image", encoding="utf-8")
    return directory


def test_parse_prompt_collects_queries_per_class() -> None:
    prompt: Prompt = ClassifyCommand.parse_prompt(["cat", "dog:dog,puppy", "dog:hound"])
    assert prompt.class_names == ("cat", "dog")
    assert prompt.queries == (TextQuery("cat"), TextQuery("dog"), TextQuery("puppy"), TextQuery("hound"))


def test_parse_prompt_reads_visual_queries(image_directory: Path) -> None:
    prompt: Prompt = ClassifyCommand.parse_prompt(
        [f"red:red,@{image_directory / 'b.png'}", f"mixed:@{image_directory}"]
    )
    assert prompt.class_names == ("red", "mixed")
    assert prompt.queries[0] == TextQuery("red")
    red_query, mixed_query = prompt.queries[1], prompt.queries[2]
    assert isinstance(red_query, VisualQuery)
    assert isinstance(mixed_query, VisualQuery)
    assert [reference.image.size for reference in red_query.references] == [(40, 30)]
    assert [reference.image.size for reference in mixed_query.references] == [(30, 40), (40, 30)]
    with pytest.raises(FileNotFoundError, match="no image file or directory"):
        ClassifyCommand.parse_prompt([f"missing:@{image_directory / 'missing.png'}"])


def test_find_images_expands_directories(image_directory: Path) -> None:
    assert [path.name for path in ImageFiles.find([image_directory], is_recursive=False)] == [
        "a.jpg",
        "b.png",
    ]
    assert [path.name for path in ImageFiles.find([image_directory], is_recursive=True)] == [
        "a.jpg",
        "b.png",
        "c.png",
    ]
    with pytest.raises(FileNotFoundError, match="no image file or directory"):
        ImageFiles.find([image_directory / "missing.png"], is_recursive=False)
    empty_directory: Path = image_directory / "empty"
    empty_directory.mkdir()
    with pytest.raises(FileNotFoundError, match="no images found"):
        ImageFiles.find([empty_directory], is_recursive=False)


def test_json_lines_output(checkpoint_directory: Path, image_directory: Path) -> None:
    output: str = run(
        [
            str(image_directory),
            "--weights",
            str(checkpoint_directory / "siglip"),
            "--backend",
            "siglip",
            "--device",
            "cpu",
            "--templates",
            "{}",
            "--classes",
            "cat",
            "dog:dog,a cat",
            "--top-k",
            "1",
            "--format",
            "jsonl",
            "--tile-grid",
            "2",
        ]
    )
    lines: list[dict[str, object]] = [json.loads(line) for line in output.splitlines()]
    assert [Path(str(line["image"])).name for line in lines] == ["a.jpg", "b.png"]
    classifications: object = lines[0]["classifications"]
    assert isinstance(classifications, list)
    assert len(classifications) == 1


def test_text_output_with_overrides(checkpoint_directory: Path, image_directory: Path) -> None:
    output: str = run(
        [
            str(image_directory / "b.png"),
            "--weights",
            str(checkpoint_directory / "clip"),
            "--backend",
            "clip",
            "--pooling",
            ImagePooling.PATCH_MEAN.value,
            "--templates",
            "{}",
            "--device",
            "cpu",
            "--classes",
            "cat",
            "dog",
            "--threshold",
            "0.0",
        ]
    )
    lines: list[str] = output.splitlines()
    assert lines[0].endswith("b.png:")
    assert len(lines) == 3


def test_list_presets() -> None:
    output: str = run(["--list-presets"])
    assert output.splitlines() == [
        f"{backend.value}/{name}" for backend in ClassifierBackend for name in PresetCatalog.names(backend)
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["image.png", "--classes", "cat"],
        ["image.png", "--classes", "cat", "--weights", "clip"],
        ["image.png", "--classes", "cat", "--preset", "clip/openai_vit_b16", "--weights", "clip", "--backend", "clip"],
        ["--preset", "clip/openai_vit_b16"],
    ],
)
def test_invalid_arguments_exit(image_directory: Path, arguments: list[str]) -> None:
    resolved: list[str] = [
        str(image_directory / "b.png") if argument == "image.png" else argument for argument in arguments
    ]
    with pytest.raises(SystemExit):
        run(resolved)


def test_unreadable_images_are_skipped(checkpoint_directory: Path, image_directory: Path) -> None:
    (image_directory / "broken.png").write_text("not an image", encoding="utf-8")
    output, warnings = run_with_warnings(
        [str(image_directory), *siglip_arguments(checkpoint_directory), "--classes", "cat", "dog"]
    )
    assert [Path(str(json.loads(line)["image"])).name for line in output.splitlines()] == ["a.jpg", "b.png"]
    assert "skipping" in warnings
    assert "broken.png" in warnings
    assert "skipped 1 of 3 image files." in warnings


def test_visual_queries_from_the_command_line(checkpoint_directory: Path, image_directory: Path) -> None:
    output: str = run(
        [
            str(image_directory / "b.png"),
            *siglip_arguments(checkpoint_directory),
            "--classes",
            "cat",
            f"red:@{image_directory / 'b.png'}",
            "--top-k",
            "2",
        ]
    )
    records: object = json.loads(output)["classifications"]
    assert isinstance(records, list)
    assert {record["matched_query_kind"] for record in records} == {"text", "visual"}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["image.png", "--classes", "cat", "--preset", "clip/openai_vit_b16", "--top-k", "0"], "--top-k"),
        (["image.png", "--classes", "cat", "--preset", "clip/openai_vit_b16", "--threshold", "2"], "--threshold"),
        (["image.png", "--classes", "cat", "--preset", "clip/openai_vit_b16", "--backend", "siglip"], "--backend"),
        (["image.png", "--classes", "cat", "--preset", "foo/bar"], "--preset"),
        (["image.png", "--classes", "cat", "--preset", "clip/missing"], "no preset"),
        (["missing.png", "--classes", "cat", "--preset", "clip/openai_vit_b16"], "no image file"),
        (["image.png", "--classes", "cat", "--preset", "clip/openai_vit_b16", "--batch-size", "0"], "batch_size"),
        (["image.png", "--classes", "cat", "--preset", "clip/openai_vit_b16", "--tile-grid", "1"], "grid_size"),
        (["image.png", "--classes", "cat", "cat", "--preset", "clip/openai_vit_b16"], "more than once"),
    ],
)
def test_invalid_arguments_are_reported_before_loading(
    image_directory: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    resolved: list[str] = [
        str(image_directory / "b.png") if argument == "image.png" else argument for argument in arguments
    ]
    with pytest.raises(SystemExit):
        run(resolved)
    assert message in capsys.readouterr().err
