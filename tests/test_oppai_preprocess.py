"""Parity tests: the ported OppaiOracle preprocessing must reproduce the
reference tensors from EagleOppaiTagger/scripts/python-ref (which are
themselves verified against the JS implementation with MAE < 5e-9).

Runs on CPU, without onnxruntime and without the model weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lorayaki.taggers.oppai_oracle import (
    letterbox,
    load_preprocessing,
    load_tag_table,
    preprocess_image,
    probs_to_tags,
)

FIXTURES = Path(__file__).parent / "fixtures"

SIZE = 448
PAD = (114, 114, 114)
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
PAD_VALUE = (114 / 255.0 - 0.5) / 0.5  # ~ -0.10196


class TestPreprocessParity:
    def test_square_full_tensor_parity(self):
        """Full-tensor comparison against the reference export (square)."""
        expected = json.loads((FIXTURES / "expected_square.json").read_text(encoding="utf-8"))
        img = Image.open(FIXTURES / "square.png")
        pixels, mask = preprocess_image(img, SIZE, PAD, MEAN, STD)

        exp_pixels = np.array(expected["pixel_values"], dtype=np.float32).reshape(3, SIZE, SIZE)
        exp_mask = np.array(expected["padding_mask"], dtype=bool).reshape(SIZE, SIZE)

        assert pixels.shape == (3, SIZE, SIZE)
        assert pixels.dtype == np.float32
        mae = np.abs(pixels - exp_pixels).mean()
        assert mae < 1e-6, f"MAE {mae} too large"
        assert np.array_equal(mask, exp_mask)

    @pytest.mark.parametrize("name", ["tall", "wide"])
    def test_letterbox_geometry(self, name: str):
        """Letterbox geometry must match the reference (arrays stripped from
        these fixtures to keep the repo small; geometry fields retained)."""
        expected = json.loads((FIXTURES / f"expected_{name}.json").read_text(encoding="utf-8"))
        img = Image.open(FIXTURES / f"{name}.png")
        _, mask = letterbox(img, SIZE, PAD)

        x0, y0 = expected["x0"], expected["y0"]
        nw, nh = expected["nw"], expected["nh"]
        # False (image) region is exactly the reference rectangle
        image_region = ~mask
        assert image_region[y0 : y0 + nh, x0 : x0 + nw].all()
        assert image_region.sum() == nw * nh

    @pytest.mark.parametrize("name", ["tall", "wide"])
    def test_pad_and_image_pixel_values(self, name: str):
        expected = json.loads((FIXTURES / f"expected_{name}.json").read_text(encoding="utf-8"))
        img = Image.open(FIXTURES / f"{name}.png")
        pixels, mask = preprocess_image(img, SIZE, PAD, MEAN, STD)

        # pad area normalized value
        pad_pixels = pixels[:, mask]
        assert pad_pixels.size > 0
        np.testing.assert_allclose(pad_pixels, PAD_VALUE, atol=1e-5)

        # center of the image region must not be the pad value
        cx = expected["x0"] + expected["nw"] // 2
        cy = expected["y0"] + expected["nh"] // 2
        center = pixels[:, cy, cx]
        assert not np.allclose(center, PAD_VALUE, atol=1e-3)
        # and must be within normalized RGB range [-1, 1]
        assert ((center >= -1.0 - 1e-6) & (center <= 1.0 + 1e-6)).all()


class TestLoadPreprocessing:
    def test_from_json(self, tmp_path: Path):
        (tmp_path / "preprocessing.json").write_text(
            json.dumps(
                {
                    "image_size": 224,
                    "pad_color_rgb": [1, 2, 3],
                    "normalize_mean": [0.1, 0.2, 0.3],
                    "normalize_std": [0.9, 0.8, 0.7],
                }
            ),
            encoding="utf-8",
        )
        pre = load_preprocessing(tmp_path)
        assert pre["image_size"] == 224
        assert pre["pad_color"] == (1, 2, 3)
        assert pre["mean"].shape == (3, 1, 1)

    def test_fallback_constants(self, tmp_path: Path):
        pre = load_preprocessing(tmp_path)  # no json at all
        assert pre["image_size"] == 448
        assert pre["pad_color"] == (114, 114, 114)


class TestTagTable:
    def test_parse_including_quoted_commas(self, tmp_path: Path):
        csv = tmp_path / "selected_tags.csv"
        csv.write_text(
            'tag_id,name,category\n0,"<PAD>",0\n1,"<UNK>",0\n2,1girl,0\n3,"breasts, large",0\n4,some_artist,1\n',
            encoding="utf-8",
        )
        names, cats = load_tag_table(csv)
        assert names == ["<PAD>", "<UNK>", "1girl", "breasts, large", "some_artist"]
        assert cats == [0, 0, 0, 0, 1]


class TestProbsToTags:
    NAMES = ["<PAD>", "<UNK>", "solo", "1girl", "artist:x", "series", "charname", "highres", "smile"]
    CATS = [0, 0, 0, 0, 1, 3, 4, 5, 0]

    def _probs(self, **overrides: float) -> np.ndarray:
        probs = np.zeros(len(self.NAMES), dtype=np.float32)
        for name, p in overrides.items():
            probs[self.NAMES.index(name)] = p
        return probs

    def test_threshold_per_category(self):
        probs = self._probs(**{"solo": 0.9, "artist:x": 0.6, "series": 0.6, "charname": 0.6, "highres": 0.3})
        tags = probs_to_tags(probs, self.NAMES, self.CATS, thresholds={0: 0.35, 1: 0.5, 3: 0.5, 4: 0.85, 5: 0.35})
        assert "solo" in tags
        assert "artist:x" in tags      # 0.6 >= 0.5
        assert "series" in tags        # 0.6 >= 0.5
        assert "charname" not in tags  # 0.6 < 0.85
        assert "highres" not in tags   # 0.3 < 0.35

    def test_drop_categories(self):
        probs = self._probs(**{"solo": 0.9, "artist:x": 0.9, "series": 0.9, "charname": 0.99})
        tags = probs_to_tags(probs, self.NAMES, self.CATS, drop_categories={1, 3, 4})
        assert tags == ["solo"]

    def test_blacklist_and_max_tags_and_order(self):
        probs = self._probs(**{"solo": 0.5, "1girl": 0.9, "smile": 0.7})
        tags = probs_to_tags(
            probs, self.NAMES, self.CATS, blacklist={"smile"}, max_tags=1
        )
        assert tags == ["1girl"]  # highest prob first, smile excluded

    def test_comma_in_tag_name_sanitized(self):
        names = ["<PAD>", "<UNK>", "breasts, large"]
        cats = [0, 0, 0]
        probs = np.array([0.0, 0.0, 0.99], dtype=np.float32)
        assert probs_to_tags(probs, names, cats) == ["breasts large"]

    def test_pad_unk_never_emitted(self):
        probs = np.ones(len(self.NAMES), dtype=np.float32)
        tags = probs_to_tags(probs, self.NAMES, self.CATS)
        assert "<PAD>" not in tags and "<UNK>" not in tags
