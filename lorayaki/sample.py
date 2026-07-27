"""Generate sample_prompts.txt for sd-scripts' in-training sampler.

Line syntax (sd-scripts/library/sampling.py):
    <prompt> --w 832 --h 1216 --s 28 --l 7 --d 42 --n <negative> --ss euler_a
"""

from __future__ import annotations

from pathlib import Path

from lorayaki.config import CharacterConfig


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def build_sample_prompt_lines(config: CharacterConfig) -> list[str]:
    samples = config.samples
    default_negative = samples.get("negative", "")
    default_sampler = samples.get("sampler", "euler_a")
    default_steps = int(samples.get("steps", 28))
    default_scale = float(samples.get("scale", 7.0))

    lines: list[str] = []
    for entry in samples.get("prompts", []):
        prompt = entry.get("prompt", "").strip()
        if not prompt:
            continue
        # trigger is part of the prompt so samples exercise the LoRA trigger
        full_prompt = f"{config.trigger}, {prompt}" if config.trigger else prompt
        w = int(entry.get("width", 832))
        h = int(entry.get("height", 1216))
        steps = int(entry.get("steps", default_steps))
        scale = float(entry.get("scale", default_scale))
        negative = entry.get("negative", default_negative)
        sampler = entry.get("sampler", default_sampler)

        line = f"{full_prompt} --w {w} --h {h} --s {steps} --l {_fmt(scale)}"
        if entry.get("seed") is not None:
            line += f" --d {int(entry['seed'])}"
        if negative:
            line += f" --n {negative}"
        line += f" --ss {sampler}"
        lines.append(line)
    return lines


def write_sample_prompts(path: Path, config: CharacterConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = build_sample_prompt_lines(config)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
