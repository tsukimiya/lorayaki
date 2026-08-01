"""WD14 tagger — invokes sd-scripts' own finetune/tag_images_by_wd14_tagger.py
inside the sd-scripts venv (no reimplementation; its onnxruntime lives there).

The script writes one <image>.txt per image containing comma-separated tags.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from lorayaki.config import GlobalConfig
from lorayaki.doctor import find_sd_scripts_python
from lorayaki.log import get_logger

WD14_SCRIPT = Path("finetune") / "tag_images_by_wd14_tagger.py"


def resolve_python(gcfg: GlobalConfig) -> str:
    """Python interpreter that runs the WD14 script (sd-scripts venv)."""
    if gcfg.sd_scripts_python:
        return gcfg.sd_scripts_python
    sd_dir = gcfg.sd_scripts_dir
    if sd_dir:
        found = find_sd_scripts_python(sd_dir)
        if found:
            return str(found)
    return "python"


def build_wd14_command(
    python: str,
    image_dir: Path,
    *,
    repo_id: str,
    onnx: bool = True,
    batch_size: int = 4,
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
    caption_extension: str = ".txt",
    remove_underscore: bool = True,
    character_tag_expand: bool = True,
    undesired_tags: list[str] | None = None,
) -> list[str]:
    cmd = [
        python,
        str(WD14_SCRIPT),
        str(image_dir),
        "--repo_id", repo_id,
        "--batch_size", str(batch_size),
        "--general_threshold", f"{general_threshold}",
        "--character_threshold", f"{character_threshold}",
        "--caption_extension", caption_extension,
    ]
    if onnx:
        cmd.append("--onnx")
    if remove_underscore:
        cmd.append("--remove_underscore")
    if character_tag_expand:
        cmd.append("--character_tag_expand")
    if undesired_tags:
        cmd += ["--undesired_tags", ",".join(undesired_tags)]
    return cmd


def run_wd14_tagging(image_dir: Path, gcfg: GlobalConfig, *, caption_extension: str = ".txt") -> None:
    """Run the WD14 tagger over *image_dir* (blocks until done)."""
    log = get_logger()
    sd_dir = gcfg.sd_scripts_dir
    if sd_dir is None or not (sd_dir / WD14_SCRIPT).exists():
        raise FileNotFoundError(
            f"WD14 タグ付けスクリプトが見つかりません: {sd_dir / WD14_SCRIPT if sd_dir else '?'} — "
            f"グローバル設定の sd_scripts_dir を確認してください"
        )
    wd = gcfg.wd14
    cmd = build_wd14_command(
        resolve_python(gcfg),
        image_dir,
        repo_id=wd.get("repo_id", "SmilingWolf/wd-v1-4-convnext-tagger-v2"),
        onnx=bool(wd.get("onnx", True)),
        batch_size=int(wd.get("batch_size", 4)),
        general_threshold=float(wd.get("general_threshold", 0.35)),
        character_threshold=float(wd.get("character_threshold", 0.85)),
        caption_extension=caption_extension,
        remove_underscore=bool(wd.get("remove_underscore", True)),
        character_tag_expand=bool(wd.get("character_tag_expand", True)),
        undesired_tags=list(wd.get("undesired_tags") or []),
    )
    log.info("WD14 実行中: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=sd_dir, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"WD14 タグ付けが失敗しました (exit {e.returncode})。"
            f"sd-scripts の venv に onnx と onnxruntime (--onnx 使用時) が入っているか確認してください。"
        ) from e
