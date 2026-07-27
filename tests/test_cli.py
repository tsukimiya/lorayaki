"""CLI smoke tests: init / new / doctor (no GPU, no network)."""

from __future__ import annotations

from pathlib import Path

import yaml

from lorayaki.cli import main


def test_init_creates_config(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["init"])
    assert rc == 0
    cfg = tmp_path / "configs" / "lorayaki.yaml"
    assert cfg.exists()
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["defaults"]["backend"] == "illustrious"

    # refuses to overwrite without --force
    assert main(["init"]) == 1
    assert main(["init", "--force"]) == 0


def test_new_scaffolds(project: Path):
    rc = main(["new", "foo"])
    assert rc == 0
    char = project / "characters" / "foo"
    assert (char / "images").is_dir()
    assert (char / "work").is_dir()
    data = yaml.safe_load((char / "character.yaml").read_text(encoding="utf-8"))
    assert data["name"] == "foo"
    assert "trigger" in data  # present (null) so the user knows to set it

    # refuses to overwrite without --force
    assert main(["new", "foo"]) == 1


def test_new_rejects_bad_name(project: Path):
    assert main(["new", "Bad Name!"]) == 1
    assert main(["new", "ümlaut"]) == 1


def test_doctor_passes_on_healthy_project(project: Path):
    rc = main(["doctor"])
    assert rc == 0


def test_doctor_fails_without_config(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 1
