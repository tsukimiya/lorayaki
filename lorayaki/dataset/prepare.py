"""Assemble the training dataset directory: copy images + captions from
characters/<name>/images/ into work/dataset/ (flat). Copies (not symlinks)
so the dataset survives mounts/rsync to the GPU machine intact."""

from __future__ import annotations

import shutil
from pathlib import Path

from lorayaki import paths
from lorayaki.config import CharacterConfig
from lorayaki.log import get_logger


def compute_repeats(
    num_images: int,
    target_steps: int = 2000,
    batch_size: int = 1,
    epochs: int = 10,
) -> int:
    """num_repeats so that total_steps = images * repeats * epochs / batch
    lands near target_steps."""
    if num_images <= 0:
        raise ValueError("num_images must be > 0")
    return max(1, round(target_steps * batch_size / (num_images * epochs)))


def prepare_dataset(
    images_dir: Path,
    dataset_dir: Path,
    caption_extension: str = ".txt",
) -> tuple[int, int]:
    """Copy images (and their captions) into *dataset_dir*, rebuilt from scratch.

    Returns (num_images, num_with_captions).
    """
    log = get_logger()
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    images = paths.list_images(images_dir)
    with_captions = 0
    for img in images:
        shutil.copy2(img, dataset_dir / img.name)
        cap = paths.caption_path_for(img, caption_extension)
        if cap.exists():
            shutil.copy2(cap, dataset_dir / cap.name)
            with_captions += 1
        else:
            log.warning("キャプションがありません (class_tokens で代替されます): %s", img.name)

    log.info("データセット組み立て: %d 枚 (%d 枚にキャプション) -> %s", len(images), with_captions, dataset_dir)
    return len(images), with_captions
