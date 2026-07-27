"""JoyCaption tagger — natural-language captions via subprocess.

Runs ``_joycaption_runner.py`` with the *joycaption venv's* python so that
torch/transformers never enter the lorayaki venv (same separation as WD14 /
sd-scripts). Mirrors :mod:`lorayaki.taggers.wd14`.

Unlike WD14 (which writes ``<image>.txt`` itself), JoyCaption returns an
in-memory ``{image_path: description}`` dict — like OppaiOracle — so the merge
step in ``tag_command`` stays the single writer of the final caption.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from lorayaki.config import GlobalConfig
from lorayaki.doctor import find_venv_python
from lorayaki.log import get_logger

RUNNER_SCRIPT = Path(__file__).parent / "_joycaption_runner.py"

# Long enough for a large dataset on one GPU (~2-5 s/image); bounds zombies.
DEFAULT_TIMEOUT = 3600


def resolve_python(gcfg: GlobalConfig) -> str:
    """Python interpreter that runs the JoyCaption runner (joycaption venv)."""
    explicit = gcfg.joycaption_python
    if explicit:
        return str(explicit)
    jc_dir = gcfg.joycaption_dir
    if jc_dir:
        found = find_venv_python(jc_dir)
        if found:
            return str(found)
    return "python"


def build_joycaption_command(
    python: str,
    *,
    images_file: Path,
    model: str,
    prompt: str,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    greedy: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> list[str]:
    cmd = [
        python,
        str(RUNNER_SCRIPT),
        "--images", str(images_file),
        "--model", model,
        "--prompt", prompt,
        "--batch-size", str(batch_size),
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", f"{temperature}",
        "--top-p", f"{top_p}",
    ]
    if greedy:
        cmd.append("--greedy")
    return cmd


def run_joycaption_tagging(images: list[Path], gcfg: GlobalConfig) -> dict[Path, str]:
    """Caption *images* with JoyCaption -> ``{image_path: description}``.

    Raises FileNotFoundError if joycaption.dir is unset/missing, RuntimeError
    if the subprocess fails.
    """
    log = get_logger()
    jc = gcfg.joycaption
    jc_dir = gcfg.joycaption_dir
    if jc_dir is None or not jc_dir.is_dir():
        raise FileNotFoundError(
            f"JoyCaption ディレクトリが見つかりません: {jc_dir} — "
            f"グローバル設定の joycaption.dir を設定してください "
            f"(JoyCaption 専用 venv が必要です)"
        )

    # One path per line avoids shell-escaping issues with spaces in paths.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for img in images:
            f.write(str(Path(img).resolve()) + "\n")
        images_file = Path(f.name)

    try:
        cmd = build_joycaption_command(
            resolve_python(gcfg),
            images_file=images_file,
            model=jc.get("model", "fancyfeast/llama-joycaption-beta-one-hf-llava"),
            prompt=jc.get("prompt", "Write a long descriptive caption for this image."),
            batch_size=int(jc.get("batch_size", 1)),
            max_new_tokens=int(jc.get("max_new_tokens", 256)),
        )
        log.info("JoyCaption 実行中: %s ...", " ".join(cmd[:6]))
        try:
            result = subprocess.run(
                cmd,
                cwd=jc_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=DEFAULT_TIMEOUT,
            )
        except subprocess.CalledProcessError as e:  # pragma: no cover - run() w/o check
            raise RuntimeError(f"JoyCaption が失敗しました: {e}") from e

        if result.returncode != 0:
            log.error("JoyCaption stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"JoyCaption が失敗しました (exit {result.returncode})。"
                f"joycaption venv に torch/transformers が入っているか確認してください。"
            )
        for line in (result.stderr or "").strip().splitlines():
            log.warning("JoyCaption: %s", line)

        try:
            raw = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JoyCaption の出力を解釈できません: {e}") from e
        return {Path(k): v for k, v in raw.items()}
    finally:
        images_file.unlink(missing_ok=True)
