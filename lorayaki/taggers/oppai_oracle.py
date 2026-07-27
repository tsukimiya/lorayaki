"""OppaiOracle tagger — in-process ONNX inference (or HTTP client fallback).

Preprocessing and model loading are ported from EagleOppaiTagger
(server/preprocess.py, server/model_loader.py, scripts/python-ref/app.py),
Apache License 2.0 — see NOTICE at the project root.

Tensor contract (OppaiOracle V1.1):
  inputs : pixel_values  float32 [N, 3, 448, 448]  (letterboxed, normalized)
           padding_mask  bool    [N, 448, 448]     (True = padded area)
  output : probabilities float   [N, 19294]        (sigmoid per-tag scores)

onnxruntime is imported lazily so the rest of the package (and the test
suite) works without it installed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from lorayaki.log import get_logger
from lorayaki.model_registry import OPPAI_V11_PREPROCESSING

# Tag categories in selected_tags.csv (danbooru convention).
CATEGORY_NAMES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta"}

DEFAULT_THRESHOLDS = {
    0: 0.35,  # general
    1: 0.5,   # artist
    3: 0.5,   # copyright
    4: 0.85,  # character
    5: 0.35,  # meta
}


# ---------------------------------------------------------------------------
# Pure preprocessing (GPU-free, unit-tested against the reference tensors)
# ---------------------------------------------------------------------------


def load_preprocessing(model_dir: Path) -> dict[str, Any]:
    """Load preprocessing.json; fall back to known V1.1 constants if absent."""
    path = model_dir / "preprocessing.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            preproc = json.load(f)
    else:
        get_logger().warning(
            "preprocessing.json がありません。V1.1 の既定定数を使用します: %s", path
        )
        preproc = dict(OPPAI_V11_PREPROCESSING)
    return {
        "image_size": int(preproc["image_size"]),
        "pad_color": tuple(int(c) for c in preproc["pad_color_rgb"]),
        "mean": np.array(preproc["normalize_mean"], dtype=np.float32).reshape(3, 1, 1),
        "std": np.array(preproc["normalize_std"], dtype=np.float32).reshape(3, 1, 1),
    }


def letterbox(
    img: Image.Image, image_size: int, pad_color: tuple[int, int, int]
) -> tuple[Image.Image, np.ndarray]:
    """Aspect-preserving resize onto a square canvas with padded-area mask.

    Line-for-line port of EagleOppaiTagger/server/preprocess.py:letterbox.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = min(image_size / w, image_size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    x0 = (image_size - nw) // 2
    y0 = (image_size - nh) // 2
    canvas.paste(resized, (x0, y0))
    mask = np.ones((image_size, image_size), dtype=bool)  # True = padded
    mask[y0 : y0 + nh, x0 : x0 + nw] = False
    return canvas, mask


def preprocess_image(
    img: Image.Image,
    image_size: int,
    pad_color: tuple[int, int, int],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Letterbox + normalize -> (pixel_values [3,H,W] float32, padding_mask [H,W] bool)."""
    canvas, mask = letterbox(img, image_size, pad_color)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = (arr - mean) / std
    return arr.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Tag table / probability -> tags
# ---------------------------------------------------------------------------


def load_tag_table(csv_path: Path) -> tuple[list[str], list[int]]:
    """Parse selected_tags.csv -> (tag_names, categories) indexed by tag_id."""
    tag_names: list[str] = []
    categories: list[int] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag_names.append(row["name"])
            categories.append(int(row["category"]))
    return tag_names, categories


def probs_to_tags(
    probs: np.ndarray,
    tag_names: Sequence[str],
    categories: Sequence[int],
    *,
    thresholds: dict[int, float] | None = None,
    default_threshold: float = 0.35,
    max_tags: int = 50,
    blacklist: set[str] | None = None,
    drop_categories: set[int] | None = None,
) -> list[str]:
    """Convert a probability vector to tag strings, in descending probability.

    - ``thresholds``: per-category minimum probability (DEFAULT_THRESHOLDS).
    - ``drop_categories``: categories excluded entirely (e.g. {1,3,4} for new
      character LoRAs — existing character/copyright/artist tags are wrong).
    - ``blacklist``: exact tag names to exclude.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    blacklist = blacklist or set()
    drop_categories = drop_categories or set()

    picks: list[tuple[float, str]] = []
    for i, p in enumerate(np.asarray(probs, dtype=np.float64)):
        if i >= len(tag_names):
            break
        name = tag_names[i]
        if name in ("<PAD>", "<UNK>"):
            continue
        cat = categories[i]
        if cat in drop_categories:
            continue
        if p < thresholds.get(cat, default_threshold):
            continue
        # Sanitize: captions are comma-separated, so a tag that contains a
        # comma (e.g. "breasts, large" in the OppaiOracle vocabulary) would
        # corrupt the caption and sd-scripts' shuffle_caption tokenization.
        name = " ".join(name.replace(",", " ").split())
        if name in blacklist:
            continue
        picks.append((float(p), name))

    picks.sort(key=lambda x: -x[0])
    return [name for _, name in picks[:max_tags]]


# ---------------------------------------------------------------------------
# Local ONNX tagger
# ---------------------------------------------------------------------------


def require_onnxruntime() -> None:
    """Fail fast with install instructions before doing expensive work
    (e.g. downloading the model) if onnxruntime is missing."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime が見つかりません。pip install -e \".[onnx]\" (GPU: .[onnx-gpu]) "
            "でインストールするか、oppai_oracle.mode: http を検討してください。"
        ) from e


def _detect_providers(preferred: str | None = None) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if preferred and preferred in available:
        return [preferred]
    for candidate in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
        if candidate in available:
            return [candidate]
    return ["CPUExecutionProvider"]


@dataclass
class TagResult:
    image: Path
    tags: list[str]
    probs: np.ndarray | None = None


class OppaiOracleTagger:
    """In-process ONNX inference against a local OppaiOracle model dir."""

    def __init__(self, model_dir: Path, provider: str | None = None):
        require_onnxruntime()
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        missing = [f for f in ("model.onnx", "selected_tags.csv") if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"OppaiOracle モデルが不完全です ({self.model_dir}): {', '.join(missing)} — "
                f"`lorayaki init --download-models` を実行してください"
            )

        pre = load_preprocessing(self.model_dir)
        self.image_size: int = pre["image_size"]
        self.pad_color: tuple[int, int, int] = pre["pad_color"]
        self.mean: np.ndarray = pre["mean"]
        self.std: np.ndarray = pre["std"]
        self.tag_names, self.categories = load_tag_table(self.model_dir / "selected_tags.csv")

        providers = _detect_providers(provider)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), sess_options=sess_opts, providers=providers
        )
        get_logger().info(
            "OppaiOracle 読み込み: %s (%d tags, provider: %s)",
            self.model_dir,
            len(self.tag_names),
            providers[0],
        )

    def preprocess(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        return preprocess_image(img, self.image_size, self.pad_color, self.mean, self.std)

    def infer(self, img: Image.Image) -> np.ndarray:
        """Single image -> probability vector [n_tags]."""
        pixels, mask = self.preprocess(img)
        out = self.session.run(
            ["probabilities"],
            {"pixel_values": pixels[np.newaxis, ...], "padding_mask": mask[np.newaxis, ...]},
        )
        return np.asarray(out[0][0], dtype=np.float32)

    def tags_for(self, probs: np.ndarray, **kwargs: Any) -> list[str]:
        return probs_to_tags(
            probs, self.tag_names, self.categories, **kwargs
        )

    def tag_images(self, images: Sequence[Path], batch_size: int = 4, **tag_kwargs: Any) -> list[TagResult]:
        """Tag a list of image paths, batching inference."""
        results: list[TagResult] = []
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            tensors = [self.preprocess(Image.open(p)) for p in batch]
            pixels = np.stack([t[0] for t in tensors]).astype(np.float32)
            masks = np.stack([t[1] for t in tensors])
            out = self.session.run(["probabilities"], {"pixel_values": pixels, "padding_mask": masks})
            probs_batch = np.asarray(out[0], dtype=np.float32)
            for path, probs in zip(batch, probs_batch):
                results.append(TagResult(image=path, tags=self.tags_for(probs, **tag_kwargs), probs=probs))
        return results


# ---------------------------------------------------------------------------
# HTTP client (talks to EagleOppaiTagger/server, POST /infer)
# ---------------------------------------------------------------------------


class OppaiOracleHTTPClient:
    """Calls a running OppaiOracle inference server.

    The server returns raw probabilities, so the tag table (selected_tags.csv)
    must still be available locally under *model_dir* for tag conversion.
    """

    def __init__(self, url: str, model_dir: Path):
        self.url = url.rstrip("/")
        self.model_dir = Path(model_dir)
        self.tag_names, self.categories = load_tag_table(self.model_dir / "selected_tags.csv")

    def infer(self, img_path: Path) -> np.ndarray:
        import requests

        with img_path.open("rb") as f:
            r = requests.post(self.url + "/infer", files={"file": (img_path.name, f)}, timeout=60)
        r.raise_for_status()
        return np.asarray(r.json()["probabilities"], dtype=np.float32)

    def tags_for(self, probs: np.ndarray, **kwargs: Any) -> list[str]:
        return probs_to_tags(probs, self.tag_names, self.categories, **kwargs)
