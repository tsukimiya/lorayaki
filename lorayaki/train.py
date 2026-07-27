"""`lorayaki train`: assemble the sd-scripts command and run it.

Runs inside the sd-scripts checkout (cwd=sd_scripts_dir) so relative module
imports in sd-scripts work exactly as documented. --dry-run prints the
assembled command without executing anything.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from lorayaki.backends import get_backend
from lorayaki.backends.base import TrainPaths
from lorayaki.config import CharacterConfig, GlobalConfig
from lorayaki.log import get_logger
from lorayaki.model_registry import resolve_model


def find_latest_state_dir(output_dir: Path) -> Path | None:
    """sd-scripts saves optimizer states as '<name>-eNNNNNN-state' / 'last-state'."""
    if not output_dir.is_dir():
        return None
    candidates = sorted(p for p in output_dir.iterdir() if p.is_dir() and p.name.endswith("-state"))
    # 'last-state' sorts before '<name>-e...' names; prefer epoch dirs, then last
    epoch_dirs = [p for p in candidates if "-state" in p.name and p.name != "last-state"]
    if epoch_dirs:
        return epoch_dirs[-1]
    return candidates[-1] if candidates else None


def run_training(args: argparse.Namespace) -> int:
    log = get_logger()
    gcfg = GlobalConfig.load(args.config)
    ccfg = CharacterConfig.load(args.name, gcfg.characters_dir)

    errors = ccfg.validate(gcfg)
    backend = get_backend(ccfg.resolve_backend(gcfg))
    errors += backend.validate_config(ccfg, gcfg)
    if errors:
        for e in errors:
            log.error("%s", e)
        return 1

    preset_name = getattr(args, "preset", None) or ccfg.resolve_preset(gcfg)
    preset = backend.presets.get(preset_name)
    if preset is None:
        log.error("プリセット '%s' がバックエンド '%s' にありません", preset_name, backend.name)
        return 1

    work = gcfg.characters_dir / args.name / "work"
    toml_path = work / "dataset.toml"
    prompts_path = work / "sample_prompts.txt"
    if not toml_path.exists() or not prompts_path.exists():
        log.info("データセット未整備のため prep を先行実行します")
        from lorayaki.prep_command import run_prep

        rc = run_prep(args)
        if rc != 0:
            return rc

    base_model = resolve_model(gcfg, ccfg.base_model)
    output_dir = work / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_dir = None
    if getattr(args, "resume", False):
        resume_dir = find_latest_state_dir(output_dir)
        if resume_dir is None:
            log.warning("--resume が指定されましたが state ディレクトリが見つかりません (%s)。最初から学習します", output_dir)

    paths_ = TrainPaths(
        base_model=base_model,
        dataset_toml=toml_path,
        sample_prompts=prompts_path,
        output_dir=output_dir,
        output_name=ccfg.name,
    )
    cmd = backend.build_train_command(
        config=ccfg, global_config=gcfg, preset=preset, paths=paths_, resume_dir=resume_dir
    )

    if getattr(args, "dry_run", False):
        print(shlex.join(cmd))
        return 0

    sd_dir = gcfg.sd_scripts_dir
    log.info("学習開始: %s (cwd=%s, preset=%s)", args.name, sd_dir, preset_name)
    log.info("cmd: %s", shlex.join(cmd))
    completed = subprocess.run(cmd, cwd=sd_dir)
    if completed.returncode != 0:
        log.error("学習が失敗しました (exit %d)", completed.returncode)
        return completed.returncode
    log.info("学習完了: %s", output_dir / f"{ccfg.name}.safetensors")
    return 0
