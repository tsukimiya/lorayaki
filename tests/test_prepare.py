"""Dataset assembly and repeats heuristic."""

from __future__ import annotations

from pathlib import Path

import pytest

from lorayaki.dataset.prepare import compute_repeats, prepare_dataset


class TestComputeRepeats:
    @pytest.mark.parametrize(
        "images,batch,expected",
        [(5, 1, 40), (10, 1, 20), (20, 1, 10), (50, 1, 4), (100, 1, 2), (500, 1, 1)],
    )
    def test_lands_near_target(self, images: int, batch: int, expected: int):
        assert compute_repeats(images, target_steps=2000, batch_size=batch, epochs=10) == expected

    def test_batch_scaling(self):
        # batch 4 needs 4x repeats for the same step count
        assert compute_repeats(20, target_steps=2000, batch_size=4, epochs=10) == 40

    def test_rejects_zero_images(self):
        with pytest.raises(ValueError):
            compute_repeats(0)


class TestPrepareDataset:
    def test_copies_images_and_captions(self, tmp_path: Path):
        images = tmp_path / "images"
        images.mkdir()
        (images / "a.png").write_bytes(b"img-a")
        (images / "a.txt").write_text("cyk girl, 1girl", encoding="utf-8")
        (images / "b.jpg").write_bytes(b"img-b")
        (images / "ignore.md").write_text("nope", encoding="utf-8")

        ds = tmp_path / "work" / "dataset"
        n, capped = prepare_dataset(images, ds)
        assert n == 2
        assert capped == 1
        assert (ds / "a.png").read_bytes() == b"img-a"
        assert (ds / "a.txt").read_text(encoding="utf-8") == "cyk girl, 1girl"
        assert (ds / "b.jpg").exists()
        assert not (ds / "ignore.md").exists()

    def test_rebuild_removes_stale_files(self, tmp_path: Path):
        images = tmp_path / "images"
        images.mkdir()
        (images / "a.png").write_bytes(b"x")
        ds = tmp_path / "dataset"
        ds.mkdir()
        (ds / "stale.png").write_bytes(b"old")

        prepare_dataset(images, ds)
        assert not (ds / "stale.png").exists()
        assert (ds / "a.png").exists()
