"""Path helpers: portability between Windows (dev) and Linux (GPU machine)."""

from __future__ import annotations

import os
from pathlib import Path

from lorayaki import paths


def test_resolve_path_absolute():
    # On Windows a path needs a drive to be absolute; on POSIX "/" suffices.
    raw = Path("C:/etc/hosts") if os.name == "nt" else Path("/etc/hosts")
    p = paths.resolve_path(str(raw))
    assert p == raw


def test_resolve_path_tilde():
    p = paths.resolve_path("~/x")
    assert p.is_absolute()
    assert "~" not in str(p)


def test_resolve_path_relative_to_base(tmp_path: Path):
    p = paths.resolve_path("models/x.onnx", base=tmp_path)
    assert p == tmp_path / "models" / "x.onnx"


def test_resolve_path_none():
    assert paths.resolve_path(None) is None


def test_caption_path_for():
    img = Path("/data/characters/foo/images/a.png")
    assert paths.caption_path_for(img) == Path("/data/characters/foo/images/a.txt")
    assert paths.caption_path_for(img, ".caption") == Path("/data/characters/foo/images/a.caption")


def test_list_images_sorted_and_filtered(tmp_path: Path):
    for name in ("b.png", "A.jpg", "c.txt", "d.WEBP", "notes.md"):
        (tmp_path / name).write_bytes(b"x")
    out = paths.list_images(tmp_path)
    assert [p.name for p in out] == ["A.jpg", "b.png", "d.WEBP"]


def test_list_images_missing_dir(tmp_path: Path):
    assert paths.list_images(tmp_path / "nope") == []


def test_character_layout(tmp_path: Path):
    root = tmp_path
    assert paths.images_dir("foo", root) == root / "characters" / "foo" / "images"
    assert paths.work_dir("foo", root) == root / "characters" / "foo" / "work"
    assert paths.dataset_toml_path("foo", root) == root / "characters" / "foo" / "work" / "dataset.toml"
    assert paths.output_dir("foo", root) == root / "characters" / "foo" / "work" / "output"
