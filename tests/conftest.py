"""Shared test fixtures. All tests run without a GPU, without network access,
and without onnxruntime — pure logic only."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake lorayaki project root as CWD, with a fake sd-scripts checkout."""
    root = tmp_path / "lorayaki"
    (root / "configs").mkdir(parents=True)
    (root / "characters").mkdir()

    sd = tmp_path / "sd-scripts"
    (sd / "finetune").mkdir(parents=True)
    (sd / "sdxl_train_network.py").write_text("# fake", encoding="utf-8")
    (sd / "anima_train_network.py").write_text("# fake", encoding="utf-8")
    (sd / "finetune" / "tag_images_by_wd14_tagger.py").write_text("# fake", encoding="utf-8")
    venv_py = sd / "venv" / "bin"
    venv_py.mkdir(parents=True)
    (venv_py / "python").write_text("# fake", encoding="utf-8")

    config = {
        "sd_scripts_dir": str(sd),
        "models": {
            "illustrious-xl-0.1": str(tmp_path / "Illustrious-XL-0.1.safetensors"),
        },
        "defaults": {"backend": "illustrious", "preset": "16gb"},
    }
    (tmp_path / "Illustrious-XL-0.1.safetensors").write_bytes(b"fake")
    with (root / "configs" / "lorayaki.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    monkeypatch.chdir(root)
    return root


@pytest.fixture
def sample_char(project: Path) -> str:
    """A character project with config + two dummy images."""
    name = "testchar"
    char_dir = project / "characters" / name
    images = char_dir / "images"
    images.mkdir(parents=True)
    for i in (1, 2):
        (images / f"img{i}.png").write_bytes(b"\x89PNG fake")
    char_cfg = {
        "name": name,
        "trigger": "cyk girl",
        "base_model": "illustrious-xl-0.1",
        "samples": {
            "prompts": [
                {"prompt": "1girl, solo", "width": 832, "height": 1216, "seed": 42},
            ]
        },
    }
    with (char_dir / "character.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(char_cfg, f, allow_unicode=True)
    return name
