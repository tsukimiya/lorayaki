"""Environment verification: `lorayaki doctor`.

Checks everything the pipeline needs, distinguishing hard errors (training
would fail) from warnings (a feature is degraded but the rest works).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from lorayaki import __version__
from lorayaki.config import GlobalConfig
from lorayaki.log import get_logger

OPPAI_REQUIRED_FILES = ("model.onnx", "selected_tags.csv")

SD_SCRIPT_VENV_CANDIDATES = (
    "venv/bin/python",
    ".venv/bin/python",
    "venv/Scripts/python.exe",
    "venv/Scripts/python",
)


def find_venv_python(directory: Path) -> Path | None:
    """Locate a python interpreter inside *directory*'s venv (common layout:
    venv/bin/python, .venv/bin/python, venv/Scripts/python.exe)."""
    for rel in SD_SCRIPT_VENV_CANDIDATES:
        candidate = directory / rel
        if candidate.exists():
            return candidate
    return None


def find_sd_scripts_python(sd_scripts_dir: Path) -> Path | None:
    """Locate the python interpreter of the sd-scripts venv."""
    return find_venv_python(sd_scripts_dir)


def _check(label: str, ok: bool | None, detail: str = "") -> bool:
    """Print one check line. ok=None means WARN. Returns True if ok (not fail)."""
    log = get_logger()
    status = "OK  " if ok else ("WARN" if ok is None else "FAIL")
    suffix = f" — {detail}" if detail else ""
    log.info("[%s] %s%s", status, label, suffix)
    return ok is not False


def run_doctor(config: GlobalConfig | None = None) -> int:
    log = get_logger()
    log.info("lorayaki %s / python %s", __version__, sys.version.split()[0])

    failures = 0

    # 1. global config ------------------------------------------------------
    try:
        config = config or GlobalConfig.load()
        _check("グローバル設定の読み込み", True, str(config.path))
    except FileNotFoundError as e:
        _check("グローバル設定の読み込み", False, str(e))
        log.info("       `lorayaki init` を実行してください。")
        return 1

    for err in config.validate():
        if not _check("グローバル設定の検証", False, err):
            failures += 1

    # 2. sd-scripts ---------------------------------------------------------
    sd_dir = config.sd_scripts_dir
    if sd_dir and sd_dir.is_dir():
        scripts = {
            "sdxl_train_network.py": "Illustrious (SDXL) 学習",
            "finetune/tag_images_by_wd14_tagger.py": "WD14 タグ付け",
            "anima_train_network.py": "Anima 学習 (将来)",
        }
        for rel, purpose in scripts.items():
            exists = (sd_dir / rel).exists()
            ok = exists if rel != "anima_train_network.py" else (True if exists else None)
            if not _check(f"sd-scripts: {rel}", ok, purpose):
                failures += 1

        py = (
            Path(config.sd_scripts_python).expanduser()
            if config.sd_scripts_python
            else find_sd_scripts_python(sd_dir)
        )
        if py and py.exists():
            _check("sd-scripts venv の python", True, str(py))
        else:
            if not _check(
                "sd-scripts venv の python",
                None,
                "venv が見つかりません。sd_scripts_python を設定してください",
            ):
                failures += 1
    elif sd_dir:
        if not _check("sd-scripts ディレクトリ", False, f"{sd_dir} が存在しません"):
            failures += 1

    # 3. model registry -----------------------------------------------------
    models = config.models
    if not models:
        _check("モデルレジストリ", None, "models が空です。ベースモデルを登録してください")
    for key, path in models.items():
        if not _check(f"モデル [{key}]", path.exists(), str(path)):
            failures += 1

    # 4. OppaiOracle --------------------------------------------------------
    mode = config.oppai_oracle_mode
    if mode == "local":
        model_dir = config.oppai_oracle_model_dir
        if model_dir and model_dir.is_dir():
            missing = [f for f in OPPAI_REQUIRED_FILES if not (model_dir / f).exists()]
            if missing:
                _check("OppaiOracle モデル", None, f"ファイル不足: {', '.join(missing)} → lorayaki init --download-models")
            else:
                _check("OppaiOracle モデル", True, str(model_dir))
        else:
            _check("OppaiOracle モデル", None, f"未ダウンロード ({model_dir}) → lorayaki init --download-models")
    elif mode == "http":
        url = config.oppai_oracle_http_url.rstrip("/") + "/health"
        try:
            import requests

            r = requests.get(url, timeout=2)
            _check("OppaiOracle サーバ", r.ok, url)
        except Exception as e:  # noqa: BLE001
            _check("OppaiOracle サーバ", None, f"{url} に接続できません ({e})")

    # 5. onnxruntime (local mode only) ---------------------------------------
    if mode == "local":
        try:
            import onnxruntime as ort  # noqa: F401

            _check("onnxruntime", True, ", ".join(ort.get_available_providers()))
        except ImportError:
            _check("onnxruntime", None, '未インストール → pip install -e ".[onnx]"')

    # 6. GPU (informational) --------------------------------------------------
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                for line in out.stdout.strip().splitlines():
                    _check("GPU", True, line.strip())
            else:
                _check("GPU", None, "nvidia-smi はあるが応答しません")
        except Exception:  # noqa: BLE001
            _check("GPU", None, "nvidia-smi 呼び出しに失敗しました")
    else:
        _check("GPU", None, "nvidia-smi が見つかりません (このマシンでは学習できません)")

    # 7. accelerate (informational) -------------------------------------------
    accel = shutil.which("accelerate")
    _check(
        "accelerate",
        True if accel else None,
        "PATH 上にあります" if accel else "PATH 上にありません (sd-scripts の venv 内なら問題ありません)",
    )

    # 8. JoyCaption (optional; only needed for the Anima backend) -------------
    jc_dir = config.joycaption_dir
    if jc_dir:
        jc_py = (
            Path(config.joycaption_python).expanduser()
            if config.joycaption_python
            else find_venv_python(jc_dir)
        )
        if jc_dir.is_dir() and jc_py and jc_py.exists():
            _check("JoyCaption", True, f"{jc_dir} (python: {jc_py})")
        elif jc_dir.is_dir():
            _check(
                "JoyCaption",
                None,
                f"venv が見つかりません: {jc_dir} (joycaption.python を設定してください)",
            )
        else:
            _check("JoyCaption", None, f"ディレクトリが存在しません: {jc_dir}")

    log.info("診断完了: %s", "問題なし" if failures == 0 else f"{failures} 件のエラー")
    return 0 if failures == 0 else 1
