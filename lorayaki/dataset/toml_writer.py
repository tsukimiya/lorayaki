"""Generate the sd-scripts dataset TOML (--dataset_config).

Schema reference: sd-scripts/docs/config_README-en.md. The legacy
"10_name"-style folder naming is deprecated in current sd-scripts — repeats
and trigger come from this file instead.
"""

from __future__ import annotations

from pathlib import Path

import tomli_w

from lorayaki.backends.base import PresetConfig
from lorayaki.config import CharacterConfig


def build_dataset_config(
    *,
    dataset_dir: Path,
    config: CharacterConfig,
    preset: PresetConfig,
    num_repeats: int,
    shuffle_caption: bool = True,
) -> dict:
    tagging = config.tagging
    return {
        "general": {
            # Disabled when captions carry a natural-language description
            # (JoyCaption / Anima): sd-scripts splits captions on commas and
            # would shuffle the sentence apart.
            "shuffle_caption": shuffle_caption,
            "caption_extension": tagging.get("caption_extension", ".txt"),
            # keep the trigger (+ always_first_tags) pinned while shuffling
            "keep_tokens": config.keep_tokens,
            "enable_bucket": True,
            "resolution": preset.resolution,
            "min_bucket_reso": preset.min_bucket_reso,
            "max_bucket_reso": preset.max_bucket_reso,
            "bucket_reso_steps": preset.bucket_reso_steps,
        },
        "datasets": [
            {
                "batch_size": preset.batch_size,
                "subsets": [
                    {
                        "image_dir": dataset_dir.as_posix(),
                        # Fallback caption when an image has no .txt file.
                        "class_tokens": config.trigger,
                        "num_repeats": num_repeats,
                        "flip_aug": False,
                        "color_aug": False,
                    }
                ],
            }
        ],
    }


def write_dataset_toml(path: Path, toml_dict: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(toml_dict, f)
    return path
