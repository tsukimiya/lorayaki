"""Model management: base-model path resolution and OppaiOracle auto-download.

OppaiOracle weights come from https://huggingface.co/Grio43/OppaiOracle
(variant V1.1_onnx). Same model as used by ../EagleOppaiTagger.
"""

from __future__ import annotations

import json
from pathlib import Path

from lorayaki.config import GlobalConfig
from lorayaki.log import get_logger

OPPAI_REPO_ID = "Grio43/OppaiOracle"
OPPAI_VARIANT = "V1.1_onnx"
OPPAI_FILES = ("model.onnx", "selected_tags.csv", "preprocessing.json")

# Fallback preprocessing constants for V1.1, verified against
# EagleOppaiTagger/scripts/python-ref/export_tensors.py (used when the HF
# repo does not ship preprocessing.json).
OPPAI_V11_PREPROCESSING = {
    "image_size": 448,
    "pad_color_rgb": [114, 114, 114],
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5],
}


def resolve_model(global_config: GlobalConfig, key: str) -> Path:
    """Resolve a model registry key to an existing path (raises otherwise)."""
    models = global_config.models
    if key not in models:
        raise KeyError(
            f"モデル '{key}' がグローバル設定の models に登録されていません "
            f"(登録済み: {', '.join(models) or 'なし'})"
        )
    path = models[key]
    if not path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {path} (キー: {key})")
    return path


def download_oppai_oracle(
    model_dir: Path | None,
    *,
    repo_id: str = OPPAI_REPO_ID,
    variant: str = OPPAI_VARIANT,
) -> Path:
    """Ensure the OppaiOracle ONNX model exists under *model_dir*.

    Downloads missing files from the HF repo. If the repo does not provide
    ``preprocessing.json``, generates it from the known V1.1 constants.
    """
    log = get_logger()
    if model_dir is None:
        model_dir = Path.cwd() / "models" / "OppaiOracle" / variant
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("huggingface_hub が必要です: pip install huggingface_hub") from e

    for filename in OPPAI_FILES:
        target = model_dir / filename
        if target.exists():
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id, filename=f"{variant}/{filename}", local_dir=str(model_dir.parent)
            )
            # hf_hub_download keeps the repo layout (<variant>/<file>); move it
            # up so the model_dir holds the files flat.
            src = Path(downloaded)
            if src != target and src.exists():
                src.replace(target)
            log.info("ダウンロード: %s", target)
        except Exception as e:  # noqa: BLE001
            if filename == "preprocessing.json":
                target.write_text(json.dumps(OPPAI_V11_PREPROCESSING, indent=2), encoding="utf-8")
                log.warning(
                    "preprocessing.json の取得に失敗したため既知の V1.1 定数から生成しました: %s (%s)",
                    target,
                    e,
                )
            else:
                raise

    missing = [f for f in ("model.onnx", "selected_tags.csv") if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"OppaiOracle モデルが不完全です ({model_dir}): {', '.join(missing)} が見つかりません")
    return model_dir
