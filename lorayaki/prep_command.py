"""`lorayaki prep`: assemble dataset dir, dataset.toml, sample_prompts.txt."""

from __future__ import annotations

import argparse

from lorayaki import paths
from lorayaki.backends import get_backend
from lorayaki.config import CharacterConfig, GlobalConfig
from lorayaki.dataset.prepare import compute_repeats, prepare_dataset
from lorayaki.dataset.toml_writer import build_dataset_config, write_dataset_toml
from lorayaki.log import get_logger
from lorayaki.sample import write_sample_prompts


def run_prep(args: argparse.Namespace) -> int:
    log = get_logger()
    gcfg = GlobalConfig.load(args.config)
    ccfg = CharacterConfig.load(args.name, gcfg.characters_dir)

    errors = ccfg.validate(gcfg)
    if errors:
        for e in errors:
            log.error("%s", e)
        return 1

    backend = get_backend(ccfg.resolve_backend(gcfg))
    preset_name = getattr(args, "preset", None) or ccfg.resolve_preset(gcfg)
    preset = backend.presets.get(preset_name)
    if preset is None:
        log.error(
            "バックエンド '%s' にはプリセット '%s' がありません (利用可: %s)",
            backend.name,
            preset_name,
            ", ".join(backend.presets) or "なし",
        )
        return 1

    images_dir = gcfg.characters_dir / args.name / "images"
    if not paths.list_images(images_dir):
        log.error("画像が見つかりません: %s", images_dir)
        return 1

    dataset_dir = gcfg.characters_dir / args.name / "work" / "dataset"
    n_images, _ = prepare_dataset(images_dir, dataset_dir, ccfg.caption_extension)

    t = ccfg.training
    epochs = int(t.get("epochs", 10))
    repeats = t.get("num_repeats")
    if repeats is None:
        repeats = compute_repeats(
            n_images,
            target_steps=int(t.get("target_steps", 2000)),
            batch_size=preset.batch_size,
            epochs=epochs,
        )
        log.info("num_repeats 自動計算: %d (目標 ~%d steps)", repeats, int(t.get("target_steps", 2000)))
    steps = n_images * repeats * epochs // preset.batch_size

    toml_path = gcfg.characters_dir / args.name / "work" / "dataset.toml"
    write_dataset_toml(
        toml_path,
        build_dataset_config(
            dataset_dir=dataset_dir,
            config=ccfg,
            preset=preset,
            num_repeats=int(repeats),
            shuffle_caption=not ccfg.joycaption_enabled(gcfg),
        ),
    )

    prompts_path = gcfg.characters_dir / args.name / "work" / "sample_prompts.txt"
    write_sample_prompts(prompts_path, ccfg)

    log.info("生成: %s", toml_path)
    log.info("生成: %s", prompts_path)
    log.info("予定: %d 枚 × repeats %d × %d epochs / batch %d ≒ %d steps",
             n_images, repeats, epochs, preset.batch_size, steps)
    return 0
