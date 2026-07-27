"""Dataset TOML generation — values must match sd-scripts' expected schema
(docs/config_README-en.md)."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore

from lorayaki.backends.illustrious import PRESETS
from lorayaki.config import CharacterConfig
from lorayaki.dataset.toml_writer import build_dataset_config, write_dataset_toml


def test_dataset_toml_structure(project: Path, sample_char: str):
    ccfg = CharacterConfig.load(sample_char)
    preset = PRESETS["16gb"]
    toml = build_dataset_config(
        dataset_dir=Path("/data/characters/testchar/work/dataset"),
        config=ccfg,
        preset=preset,
        num_repeats=10,
    )
    g = toml["general"]
    assert g["shuffle_caption"] is True
    assert g["caption_extension"] == ".txt"
    assert g["keep_tokens"] == 1  # trigger only
    assert g["enable_bucket"] is True
    assert g["resolution"] == 1024
    assert g["bucket_reso_steps"] == 64
    assert g["min_bucket_reso"] == 512
    assert g["max_bucket_reso"] == 1536

    ds = toml["datasets"][0]
    assert ds["batch_size"] == 2
    subset = ds["subsets"][0]
    assert subset["image_dir"] == "/data/characters/testchar/work/dataset"  # posix
    assert subset["class_tokens"] == "cyk girl"  # fallback caption = trigger
    assert subset["num_repeats"] == 10
    assert subset["flip_aug"] is False


def test_keep_tokens_includes_always_first(project: Path, sample_char: str):
    ccfg = CharacterConfig.load(sample_char)
    ccfg.data["tagging"]["always_first_tags"] = ["1girl", "solo"]
    toml = build_dataset_config(
        dataset_dir=Path("/x"), config=ccfg, preset=PRESETS["12gb"], num_repeats=5
    )
    assert toml["general"]["keep_tokens"] == 3


def test_write_and_roundtrip(project: Path, sample_char: str, tmp_path: Path):
    ccfg = CharacterConfig.load(sample_char)
    toml = build_dataset_config(
        dataset_dir=tmp_path / "ds", config=ccfg, preset=PRESETS["16gb"], num_repeats=7
    )
    out = write_dataset_toml(tmp_path / "dataset.toml", toml)
    parsed = tomllib.loads(out.read_text(encoding="utf-8"))
    assert parsed["datasets"][0]["subsets"][0]["num_repeats"] == 7
    assert parsed["general"]["keep_tokens"] == 1
