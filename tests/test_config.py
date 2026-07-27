"""Config loading, merging, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lorayaki.config import (
    CHARACTER_DEFAULTS,
    GLOBAL_DEFAULTS,
    CharacterConfig,
    GlobalConfig,
    character_template,
    deep_merge,
)


class TestDeepMerge:
    def test_user_wins(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        out = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
        assert out == {"a": {"b": 9, "c": 2}}

    def test_defaults_untouched(self):
        defaults = {"a": {"b": 1}}
        deep_merge(defaults, {"a": {"b": 2}})
        assert defaults == {"a": {"b": 1}}

    def test_new_keys_added(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


class TestGlobalConfig:
    def test_load_and_accessors(self, project: Path):
        cfg = GlobalConfig.load()
        assert cfg.path is not None and cfg.path.exists()
        assert cfg.sd_scripts_dir.name == "sd-scripts"
        assert cfg.sd_scripts_dir.is_absolute()
        assert "illustrious-xl-0.1" in cfg.models
        assert cfg.default_backend == "illustrious"
        assert cfg.default_preset == "16gb"

    def test_defaults_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        cfg = GlobalConfig.load(required=False)
        assert cfg.default_backend == "illustrious"
        assert cfg.oppai_oracle_mode == "local"

    def test_missing_required_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            GlobalConfig.load(required=True)

    def test_validate_ok(self, project: Path):
        cfg = GlobalConfig.load()
        assert cfg.validate() == []

    def test_validate_bad_sd_scripts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        cfg = GlobalConfig(data=deep_merge(GLOBAL_DEFAULTS, {"sd_scripts_dir": "/nonexistent"}))
        errors = cfg.validate()
        assert any("sd_scripts_dir" in e for e in errors)


class TestCharacterConfig:
    def test_load_merges_defaults(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        assert cfg.trigger == "cyk girl"
        # merged from defaults
        assert cfg.training["epochs"] == 10
        assert cfg.training["optimizer"] == "AdamW8bit"
        assert cfg.network["dim"] is None  # preset decides later
        assert cfg.caption_extension == ".txt"

    def test_keep_tokens(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        assert cfg.keep_tokens == 1  # trigger only
        cfg.data["tagging"]["always_first_tags"] = ["1girl", "solo"]
        assert cfg.keep_tokens == 3

    def test_validate_missing_trigger(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        cfg.data["trigger"] = None
        errors = cfg.validate()
        assert any("trigger" in e for e in errors)

    def test_validate_base_model_not_registered(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        cfg.data["base_model"] = "does-not-exist"
        gcfg = GlobalConfig.load()
        errors = cfg.validate(gcfg)
        assert any("base_model" in e for e in errors)

    def test_validate_dim_without_alpha(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        cfg.data["network"]["dim"] = 32
        cfg.data["network"]["alpha"] = None
        errors = cfg.validate()
        assert any("alpha" in e for e in errors)

    def test_resolve_backend_and_preset(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        gcfg = GlobalConfig.load()
        assert cfg.resolve_backend(gcfg) == "illustrious"
        assert cfg.resolve_preset(gcfg) == "16gb"
        cfg.data["training"]["preset"] = "24gb"
        assert cfg.resolve_preset(gcfg) == "24gb"

    def test_load_missing_raises(self, project: Path):
        with pytest.raises(FileNotFoundError):
            CharacterConfig.load("nope")

    def test_template_roundtrip(self):
        """character_template() must merge cleanly over CHARACTER_DEFAULTS and
        produce a config with all expected sections."""
        merged = deep_merge(CHARACTER_DEFAULTS, character_template())
        assert merged["training"]["optimizer"] == "AdamW8bit"
        assert merged["tagging"]["drop_categories"] == [1, 3, 4]
        assert merged["samples"]["steps"] == 28

    def test_save_reload(self, project: Path, sample_char: str):
        cfg = CharacterConfig.load(sample_char)
        cfg.data["training"]["epochs"] = 25
        cfg.save()
        again = CharacterConfig.load(sample_char)
        assert again.training["epochs"] == 25
        # user's original value preserved
        assert again.trigger == "cyk girl"
