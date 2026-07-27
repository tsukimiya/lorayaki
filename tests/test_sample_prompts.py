"""Sample prompt file — syntax per sd-scripts/library/sampling.py."""

from __future__ import annotations

from pathlib import Path

from lorayaki.config import CharacterConfig
from lorayaki.sample import build_sample_prompt_lines, write_sample_prompts


def test_line_syntax(project: Path, sample_char: str):
    ccfg = CharacterConfig.load(sample_char)
    lines = build_sample_prompt_lines(ccfg)
    assert len(lines) == 1
    line = lines[0]
    # trigger is prepended so samples exercise the LoRA
    assert line.startswith("cyk girl, 1girl, solo ")
    assert " --w 832" in line
    assert " --h 1216" in line
    assert " --s 28" in line
    assert " --l 7" in line
    assert " --d 42" in line
    assert " --n " in line
    assert line.endswith(" --ss euler_a")


def test_defaults_from_samples_section(project: Path, sample_char: str):
    ccfg = CharacterConfig.load(sample_char)
    ccfg.data["samples"]["prompts"] = [{"prompt": "1girl"}]  # no size/seed
    lines = build_sample_prompt_lines(ccfg)
    line = lines[0]
    assert " --w 832 --h 1216" in line        # defaults
    assert " --s 28" in line and " --l 7" in line
    assert " --d " not in line                # seed omitted when unset
    assert " --ss euler_a" in line


def test_per_entry_overrides(project: Path, sample_char: str):
    ccfg = CharacterConfig.load(sample_char)
    ccfg.data["samples"]["prompts"] = [
        {"prompt": "1girl", "width": 1024, "height": 1024, "steps": 20,
         "scale": 5.5, "seed": 7, "negative": "bad", "sampler": "ddim"}
    ]
    line = build_sample_prompt_lines(ccfg)[0]
    assert " --w 1024 --h 1024 --s 20 --l 5.5 --d 7 --n bad --ss ddim" in line


def test_write(project: Path, sample_char: str, tmp_path: Path):
    ccfg = CharacterConfig.load(sample_char)
    out = write_sample_prompts(tmp_path / "sample_prompts.txt", ccfg)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert content.count("\n") == 1  # one prompt + trailing newline
