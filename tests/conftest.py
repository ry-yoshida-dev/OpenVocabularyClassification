import json
from pathlib import Path

import pytest
import torch
from transformers import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    GemmaTokenizer,
    MetaClip2Config,
    MetaClip2Model,
    Siglip2Config,
    Siglip2ImageProcessor,
    Siglip2Model,
    Siglip2Processor,
    SiglipConfig,
    SiglipImageProcessor,
    SiglipModel,
    SiglipProcessor,
)

VISION_CONFIG: dict[str, int] = {
    "hidden_size": 32,
    "intermediate_size": 37,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "image_size": 32,
    "patch_size": 8,
}
TEXT_CONFIG: dict[str, int] = {
    "vocab_size": 16,
    "hidden_size": 32,
    "intermediate_size": 37,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 12,
}
CLIP_TEXT_CONFIG: dict[str, int] = {**TEXT_CONFIG, "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1}
CLIP_VOCABULARY: dict[str, int] = {
    "<|startoftext|>": 0,
    "<|endoftext|>": 1,
    "a</w>": 2,
    "c": 3,
    "a": 4,
    "t</w>": 5,
    "ca": 6,
    "cat</w>": 7,
    "d": 8,
    "o": 9,
    "g</w>": 10,
    "do": 11,
    "dog</w>": 12,
}
CLIP_MERGES: list[str] = ["c a", "ca t</w>", "d o", "do g</w>"]
GEMMA_VOCABULARY: dict[str, int] = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "a": 5,
    "c": 6,
    "t": 7,
    "ca": 8,
    "cat": 9,
    "d": 10,
    "o": 11,
    "g": 12,
    "do": 13,
    "dog": 14,
    "▁": 15,
}
GEMMA_MERGES: list[str] = ["c a", "ca t", "d o", "do g"]


def write_vocabulary(directory: Path, vocabulary: dict[str, int], merges: list[str]) -> tuple[str, str]:
    directory.mkdir(parents=True)
    vocabulary_path: Path = directory / "vocab.json"
    merges_path: Path = directory / "merges.txt"
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    merges_path.write_text("\n".join(merges) + "\n", encoding="utf-8")
    return str(vocabulary_path), str(merges_path)


def clip_processor(vocabulary_directory: Path) -> CLIPProcessor:
    vocabulary_path, merges_path = write_vocabulary(vocabulary_directory, CLIP_VOCABULARY, CLIP_MERGES)
    return CLIPProcessor(
        image_processor=CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
        tokenizer=CLIPTokenizer(vocab=vocabulary_path, merges=merges_path),
    )


def gemma_tokenizer(vocabulary_directory: Path) -> GemmaTokenizer:
    vocabulary_path, merges_path = write_vocabulary(vocabulary_directory, GEMMA_VOCABULARY, GEMMA_MERGES)
    return GemmaTokenizer(vocab=vocabulary_path, merges=merges_path)


@pytest.fixture(scope="session")
def checkpoint_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory: Path = tmp_path_factory.mktemp("checkpoints")
    torch.manual_seed(0)
    CLIPModel(CLIPConfig(vision_config=VISION_CONFIG, text_config=CLIP_TEXT_CONFIG, projection_dim=16)).save_pretrained(
        directory / "clip"
    )
    clip_processor(directory / "clip_vocabulary").save_pretrained(directory / "clip")
    MetaClip2Model(
        MetaClip2Config(vision_config=VISION_CONFIG, text_config=CLIP_TEXT_CONFIG, projection_dim=16)
    ).save_pretrained(directory / "metaclip_2")
    clip_processor(directory / "metaclip_2_vocabulary").save_pretrained(directory / "metaclip_2")
    SiglipModel(SiglipConfig(vision_config=VISION_CONFIG, text_config=TEXT_CONFIG)).save_pretrained(
        directory / "siglip"
    )
    SiglipProcessor(
        image_processor=SiglipImageProcessor(size={"height": 32, "width": 32}),
        tokenizer=gemma_tokenizer(directory / "siglip_vocabulary"),
    ).save_pretrained(directory / "siglip")
    Siglip2Model(
        Siglip2Config(vision_config={**VISION_CONFIG, "num_patches": 16}, text_config=TEXT_CONFIG)
    ).save_pretrained(directory / "siglip2")
    Siglip2Processor(
        image_processor=Siglip2ImageProcessor(patch_size=8, max_num_patches=16),
        tokenizer=gemma_tokenizer(directory / "siglip2_vocabulary"),
    ).save_pretrained(directory / "siglip2")
    return directory
