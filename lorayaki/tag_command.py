"""`lorayaki tag`: run OppaiOracle + WD14, merge, write captions.

Captions are written next to each image as <image>.txt so the user can review
and hand-edit them before `prep` assembles the dataset. Re-running without
--force skips images that already have a caption.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lorayaki import paths
from lorayaki.config import CharacterConfig, GlobalConfig
from lorayaki.log import get_logger
from lorayaki.taggers.merge import caption_to_tags, merge_tags, tags_to_caption
from lorayaki.taggers.oppai_oracle import DEFAULT_THRESHOLDS, OppaiOracleHTTPClient, OppaiOracleTagger


def _oppai_thresholds(tagging_cfg: dict) -> dict[int, float]:
    o = tagging_cfg.get("oppai_oracle", {})
    return {
        0: float(o.get("threshold_general", DEFAULT_THRESHOLDS[0])),
        1: float(o.get("threshold_artist", DEFAULT_THRESHOLDS[1])),
        3: float(o.get("threshold_copyright", DEFAULT_THRESHOLDS[3])),
        4: float(o.get("threshold_character", DEFAULT_THRESHOLDS[4])),
        5: float(o.get("threshold_meta", DEFAULT_THRESHOLDS[5])),
    }


def _oppai_tag_kwargs(tagging_cfg: dict) -> dict:
    o = tagging_cfg.get("oppai_oracle", {})
    return {
        "thresholds": _oppai_thresholds(tagging_cfg),
        "max_tags": int(o.get("max_tags", 60)),
        "drop_categories": set(int(c) for c in tagging_cfg.get("drop_categories", [])),
        "blacklist": set(tagging_cfg.get("extra_remove_tags", [])),
    }


def _run_oppai(
    todo: list[Path], gcfg: GlobalConfig, ccfg: CharacterConfig
) -> dict[Path, list[str]]:
    """OppaiOracle over *todo* images -> {image_path: tags}."""
    log = get_logger()
    tagging = ccfg.tagging
    kwargs = _oppai_tag_kwargs(tagging)
    mode = gcfg.oppai_oracle_mode
    results: dict[Path, list[str]] = {}

    if mode == "http":
        model_dir = gcfg.oppai_oracle_model_dir
        if model_dir is None or not (model_dir / "selected_tags.csv").exists():
            raise FileNotFoundError(
                f"http モードでもタグ一覧 (selected_tags.csv) が必要です: {model_dir} — "
                f"`lorayaki init --download-models` を実行してください"
            )
        client = OppaiOracleHTTPClient(gcfg.oppai_oracle_http_url, model_dir)
        for p in todo:
            probs = client.infer(p)
            results[p] = client.tags_for(probs, **kwargs)
        return results

    # local: fail fast if onnxruntime is missing, BEFORE downloading weights
    from lorayaki.taggers.oppai_oracle import require_onnxruntime

    require_onnxruntime()
    # ensure weights are present (auto-download on first use)
    from lorayaki.model_registry import download_oppai_oracle

    model_dir = download_oppai_oracle(gcfg.oppai_oracle_model_dir)
    tagger = OppaiOracleTagger(model_dir, provider=gcfg.oppai_oracle_provider)
    for r in tagger.tag_images(todo, **kwargs):
        results[r.image] = r.tags
    return results


def _run_wd14(images_dir: Path, gcfg: GlobalConfig, caption_extension: str) -> None:
    from lorayaki.taggers.wd14 import run_wd14_tagging

    run_wd14_tagging(images_dir, gcfg, caption_extension=caption_extension)


def run_tag(args: argparse.Namespace) -> int:
    log = get_logger()
    gcfg = GlobalConfig.load(args.config)
    ccfg = CharacterConfig.load(args.name, gcfg.characters_dir)

    errors = [e for e in ccfg.validate(gcfg) if "trigger" in e]
    if errors:
        for e in errors:
            log.error("%s", e)
        return 1

    images_dir = gcfg.characters_dir / args.name / "images"
    images = paths.list_images(images_dir)
    if not images:
        log.error("画像が見つかりません: %s", images_dir)
        return 1

    ext = ccfg.caption_extension
    todo = [
        img for img in images if args.force or not paths.caption_path_for(img, ext).exists()
    ]
    log.info(
        "%s: 画像 %d 枚、タグ付け対象 %d 枚%s",
        args.name,
        len(images),
        len(todo),
        "" if args.force else " (既存キャプションはスキップ、--force で再実行)",
    )
    if not todo:
        return 0

    tagging = ccfg.tagging
    oppai_enabled = tagging.get("oppai_oracle", {}).get("enabled", True) and not getattr(args, "wd14_only", False)
    wd14_enabled = tagging.get("wd14", {}).get("enabled", True) and not getattr(args, "oppai_only", False)

    # 1. OppaiOracle (in-memory results)
    oppai_tags: dict[Path, list[str]] = {}
    if oppai_enabled:
        try:
            oppai_tags = _run_oppai(todo, gcfg, ccfg)
        except Exception as e:  # noqa: BLE001
            log.error("OppaiOracle タグ付けに失敗: %s", e)
            return 1

    # 2. WD14 (writes .txt files itself; runs over the whole images dir)
    if wd14_enabled:
        try:
            _run_wd14(images_dir, gcfg, ext)
        except Exception as e:  # noqa: BLE001
            log.error("WD14 タグ付けに失敗: %s", e)
            return 1

    # 3. Merge and write final captions for the todo set
    extra_remove = set(tagging.get("extra_remove_tags", []))
    counter: Counter[str] = Counter()
    for img in todo:
        wd14_tags = None
        cap_file = paths.caption_path_for(img, ext)
        if wd14_enabled and cap_file.exists():
            wd14_tags = caption_to_tags(cap_file.read_text(encoding="utf-8"))
        merged = merge_tags(
            oppai_tags.get(img),
            wd14_tags,
            trigger=ccfg.trigger,
            always_first_tags=ccfg.always_first_tags,
            extra_remove_tags=extra_remove,
        )
        cap_file.write_text(tags_to_caption(merged), encoding="utf-8")
        counter.update(merged)

    avg = sum(len(caption_to_tags(p.read_text(encoding="utf-8"))) for p in
              (paths.caption_path_for(i, ext) for i in todo)) / len(todo)
    log.info("キャプション生成完了: %d 枚 (平均 %.1f タグ)", len(todo), avg)
    top = ", ".join(f"{t}({n})" for t, n in counter.most_common(10))
    log.info("頻出タグ: %s", top)
    return 0
