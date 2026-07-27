"""Command-line interface: single `lorayaki` entry with subcommands.

    init    create global config (+ optional OppaiOracle model download)
    new     scaffold a character project
    tag     tag images (OppaiOracle + WD14) and write captions
    prep    assemble dataset dir + dataset.toml + sample_prompts.txt
    train   run training via sd-scripts (--dry-run prints the command)
    run     tag -> prep -> train in one shot
    doctor  verify the environment
"""

from __future__ import annotations

import argparse
import importlib.resources
import re
import sys
from pathlib import Path

from lorayaki import __version__
from lorayaki.config import CharacterConfig, GlobalConfig, character_template
from lorayaki.log import get_logger, setup_logging
from lorayaki import paths

CHAR_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

GLOBAL_TEMPLATE_RESOURCE = "templates/global_config.yaml"


def _load_global(args: argparse.Namespace, *, required: bool = True) -> GlobalConfig:
    return GlobalConfig.load(getattr(args, "config", None), required=required)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    log = get_logger()
    target = Path(args.config) if args.config else Path.cwd() / "configs" / "lorayaki.yaml"
    if target.exists() and not args.force:
        log.error("設定が既に存在します: %s (--force で上書き)", target)
        return 1
    template = (
        importlib.resources.files("lorayaki")
        .joinpath(GLOBAL_TEMPLATE_RESOURCE)
        .read_text(encoding="utf-8")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template, encoding="utf-8")
    log.info("グローバル設定を作成しました: %s", target)
    log.info("次のステップ:")
    log.info("  1. %s を開き sd_scripts_dir と models を設定", target)
    log.info("  2. lorayaki doctor        # 環境チェック")
    log.info("  3. lorayaki init --download-models   # OppaiOracle モデル取得")

    if args.download_models:
        from lorayaki.model_registry import download_oppai_oracle

        config = GlobalConfig.load(args.config, required=False)
        try:
            path = download_oppai_oracle(config.oppai_oracle_model_dir)
            log.info("OppaiOracle モデルを用意しました: %s", path)
        except Exception as e:  # noqa: BLE001
            log.error("OppaiOracle モデルの取得に失敗: %s", e)
            return 1
    return 0


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------


def cmd_new(args: argparse.Namespace) -> int:
    log = get_logger()
    name = args.name
    if not CHAR_NAME_RE.match(name):
        log.error("キャラ名は半角小文字・数字・-_ のみ (先頭は英数字): %r", name)
        return 1

    config = _load_global(args, required=False)
    root = config.characters_dir
    char_dir = root / name
    images = char_dir / "images"
    cfg_path = char_dir / "character.yaml"

    if cfg_path.exists() and not args.force:
        log.error("既に存在します: %s (--force で上書き)", cfg_path)
        return 1

    images.mkdir(parents=True, exist_ok=True)
    (char_dir / "work").mkdir(parents=True, exist_ok=True)

    data = character_template()
    data["name"] = name
    import yaml

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        f.write(
            f"# キャラ '{name}' の設定。trigger と base_model は必須です。\n"
            f"# 他は省略可能 — 省略時は既定値 (lorayaki/config.py) が使われます。\n\n"
        )
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    log.info("キャラ雛形を作成しました: %s", char_dir)
    log.info("次のステップ:")
    log.info("  1. %s に画像を投入", images)
    log.info("  2. %s の trigger と base_model を設定", cfg_path)
    log.info("  3. lorayaki run %s", name)
    return 0


# ---------------------------------------------------------------------------
# tag / prep / train / run  (wired up in their implementation phases)
# ---------------------------------------------------------------------------


def cmd_tag(args: argparse.Namespace) -> int:
    from lorayaki.tag_command import run_tag

    return run_tag(args)


def cmd_prep(args: argparse.Namespace) -> int:
    from lorayaki.prep_command import run_prep

    return run_prep(args)


def cmd_train(args: argparse.Namespace) -> int:
    from lorayaki.train import run_training

    return run_training(args)


def cmd_run(args: argparse.Namespace) -> int:
    rc = cmd_tag(args)
    if rc != 0:
        return rc
    rc = cmd_prep(args)
    if rc != 0:
        return rc
    return cmd_train(args)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from lorayaki.doctor import run_doctor

    try:
        config = _load_global(args, required=False)
    except Exception as e:  # noqa: BLE001
        get_logger().error("設定の読み込みに失敗: %s", e)
        return 1
    return run_doctor(config)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lorayaki",
        description="新キャラ用 LoRA を素早く焼くパイプライン (sd-scripts ベース)",
    )
    parser.add_argument("--version", action="version", version=f"lorayaki {__version__}")
    parser.add_argument("--config", help="グローバル設定のパス (既定: ./configs/lorayaki.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細ログ")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="グローバル設定を作成")
    p.add_argument("--force", action="store_true", help="既存の設定を上書き")
    p.add_argument("--download-models", action="store_true", help="OppaiOracle モデルも取得")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("new", help="キャラ雛形を作成")
    p.add_argument("name", help="キャラ名 (英小文字・数字・-_)")
    p.add_argument("--force", action="store_true", help="既存を上書き")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("tag", help="画像をタグ付けしてキャプションを生成")
    p.add_argument("name", help="キャラ名")
    p.add_argument("--force", action="store_true", help="既存キャプションも再タグ付け")
    p.add_argument("--oppai-only", action="store_true", help="OppaiOracle のみ実行")
    p.add_argument("--wd14-only", action="store_true", help="WD14 のみ実行")
    p.add_argument("--joycaption-only", action="store_true", help="JoyCaption のみ実行")
    p.set_defaults(func=cmd_tag)

    p = sub.add_parser("prep", help="データセットと設定ファイルを組み立て")
    p.add_argument("name", help="キャラ名")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("train", help="学習を実行 (sd-scripts)")
    p.add_argument("name", help="キャラ名")
    p.add_argument("--dry-run", action="store_true", help="コマンドを表示するだけで実行しない")
    p.add_argument("--preset", choices=["12gb", "16gb", "24gb"], help="VRAM プリセット上書き")
    p.add_argument("--resume", action="store_true", help="最新の state から再開")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="tag -> prep -> train を一括実行")
    p.add_argument("name", help="キャラ名")
    p.add_argument("--force", action="store_true", help="既存キャプションも再タグ付け")
    p.add_argument("--dry-run", action="store_true", help="学習コマンドを表示するだけで実行しない")
    p.add_argument("--preset", choices=["12gb", "16gb", "24gb"], help="VRAM プリセット上書き")
    p.add_argument("--resume", action="store_true", help="最新の state から再開")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("doctor", help="環境診断")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except FileNotFoundError as e:
        get_logger().error("%s", e)
        return 1
    except KeyboardInterrupt:
        get_logger().error("中断しました")
        return 130


if __name__ == "__main__":
    sys.exit(main())
