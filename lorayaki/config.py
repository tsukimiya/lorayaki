"""Configuration loading, defaults, and validation.

Two config files:

- **Global** (``configs/lorayaki.yaml``): machine-specific — sd-scripts
  location, model registry, tagger settings, default backend/preset.
- **Per-character** (``characters/<name>/character.yaml``): character-specific
  — trigger word, base model, network/training/tagging/sample settings.

Both are plain YAML dicts merged over ``GLOBAL_DEFAULTS`` /
``CHARACTER_DEFAULTS`` (deep merge; user values win). Wrapping classes expose
typed accessors but keep the raw dict available for pass-through args.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lorayaki import paths

# ---------------------------------------------------------------------------
# Defaults — single source of truth for schema, templates, and merging
# ---------------------------------------------------------------------------

GLOBAL_DEFAULTS: dict[str, Any] = {
    # Required: absolute path to a kohya sd-scripts checkout with a working venv.
    "sd_scripts_dir": None,
    # Optional: python interpreter of the sd-scripts venv. Auto-detected when null.
    "sd_scripts_python": None,
    # Where character projects live (relative to CWD unless absolute).
    "characters_dir": "characters",
    # Model registry: logical key -> path to .safetensors (or diffusers dir).
    # e.g. {"illustrious-xl-0.1": "/models/Illustrious-XL-0.1.safetensors"}
    "models": {},
    "defaults": {
        "backend": "illustrious",  # illustrious | anima
        "preset": "16gb",          # 12gb | 16gb | 24gb
    },
    "oppai_oracle": {
        "mode": "local",           # local (in-process onnxruntime) | http
        "model_dir": "models/OppaiOracle/V1.1_onnx",
        "http_url": "http://localhost:8765",
        "provider": None,          # null = auto (CUDA/DirectML/CPU)
    },
    "wd14": {
        "repo_id": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "onnx": True,
        "batch_size": 4,
        "general_threshold": 0.35,
        "character_threshold": 0.85,
        "remove_underscore": True,
        "character_tag_expand": True,
        "undesired_tags": [],
    },
}

CHARACTER_DEFAULTS: dict[str, Any] = {
    "name": None,
    # Required: trigger token(s) prepended to every caption, e.g. "cyk girl".
    "trigger": None,
    # null -> global defaults.backend
    "backend": None,
    # Required: key into the global model registry.
    "base_model": None,
    # null fields fall back to the VRAM preset at train time.
    "network": {
        "dim": None,
        "alpha": None,
        "conv_dim": None,   # conv_* set => LoCon (Conv2d 3x3 targets enabled)
        "conv_alpha": None,
    },
    "training": {
        "epochs": 10,
        "num_repeats": None,      # null -> auto from target_steps
        "target_steps": 2000,
        "batch_size": None,       # null -> preset
        "preset": None,           # null -> global defaults.preset
        "unet_lr": 1e-4,
        "optimizer": "AdamW8bit",
        "scheduler": "cosine_with_restarts",
        "scheduler_args": {"num_cycles": 3},
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "network_train_unet_only": True,
        "xformers": True,
        "max_grad_norm": 1.0,
        "save_every_n_epochs": 1,
        "sample_every_n_epochs": 1,
        "seed": None,
        # Backend-specific pass-through, rendered as --key value flags.
        "extra_args": {},
    },
    "tagging": {
        "caption_extension": ".txt",
        "oppai_oracle": {
            "enabled": True,
            "threshold_general": 0.35,
            "threshold_meta": 0.35,
            "threshold_character": 0.85,
            "threshold_copyright": 0.5,
            "threshold_artist": 0.5,
            "max_tags": 60,
        },
        # WD14 thresholds come from the global config; this only toggles it.
        "wd14": {
            "enabled": True,
        },
        # Tag categories dropped before merging (new characters get wrong
        # existing-character/copyright/artist tags): 1=artist 3=copyright 4=character.
        "drop_categories": [1, 3, 4],
        "extra_remove_tags": [
            "watermark",
            "signature",
            "username",
            "text",
            "jpeg artifacts",
        ],
        # Tags pinned right after the trigger (kept in place by keep_tokens).
        "always_first_tags": [],
    },
    "samples": {
        "negative": (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, signature"
        ),
        "sampler": "euler_a",
        "steps": 28,
        "scale": 7.0,
        "prompts": [
            {"prompt": "1girl, solo, looking at viewer, smile", "width": 832, "height": 1216, "seed": 42},
            {"prompt": "1girl, solo, portrait, closeup", "width": 1024, "height": 1024, "seed": 13},
        ],
    },
}

KNOWN_BACKENDS = ("illustrious", "anima")
KNOWN_PRESETS = ("12gb", "16gb", "24gb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deep_merge(defaults: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *user* over a copy of *defaults* (user wins)."""
    out = copy.deepcopy(defaults)
    for key, value in (user or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: トップレベルは mapping (key: value) である必要があります")
    return data


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------


def find_global_config(explicit: str | Path | None = None) -> Path | None:
    """Resolve the global config path: --config > env > ./configs/lorayaki.yaml."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("LORA_BUILDER_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    default = Path.cwd() / "configs" / "lorayaki.yaml"
    return default.resolve() if default.exists() else None


@dataclass
class GlobalConfig:
    data: dict[str, Any]
    path: Path | None = None  # file it was loaded from (None = defaults only)

    @classmethod
    def load(cls, explicit: str | Path | None = None, *, required: bool = True) -> "GlobalConfig":
        path = find_global_config(explicit)
        if path is None or not path.exists():
            if required:
                raise FileNotFoundError(
                    "グローバル設定が見つかりません。`lorayaki init` を実行するか "
                    "--config で指定してください。"
                )
            return cls(data=copy.deepcopy(GLOBAL_DEFAULTS), path=None)
        data = deep_merge(GLOBAL_DEFAULTS, _load_yaml(path))
        return cls(data=data, path=path)

    def save(self, path: Path | None = None) -> Path:
        target = path or self.path
        if target is None:
            target = Path.cwd() / "configs" / "lorayaki.yaml"
        _dump_yaml(target, self.data)
        self.path = target
        return target

    # -- typed accessors ----------------------------------------------------

    @property
    def sd_scripts_dir(self) -> Path | None:
        return paths.resolve_path(self.data.get("sd_scripts_dir"))

    @property
    def sd_scripts_python(self) -> str | None:
        return self.data.get("sd_scripts_python")

    @property
    def characters_dir(self) -> Path:
        return paths.resolve_path(self.data.get("characters_dir", "characters")) or (
            Path.cwd() / "characters"
        )

    @property
    def models(self) -> dict[str, Path]:
        return {
            key: paths.resolve_path(value)
            for key, value in (self.data.get("models") or {}).items()
            if value
        }

    @property
    def default_backend(self) -> str:
        return self.data.get("defaults", {}).get("backend", "illustrious")

    @property
    def default_preset(self) -> str:
        return self.data.get("defaults", {}).get("preset", "16gb")

    @property
    def oppai_oracle_model_dir(self) -> Path | None:
        return paths.resolve_path(self.data.get("oppai_oracle", {}).get("model_dir"))

    @property
    def oppai_oracle_mode(self) -> str:
        return self.data.get("oppai_oracle", {}).get("mode", "local")

    @property
    def oppai_oracle_http_url(self) -> str:
        return self.data.get("oppai_oracle", {}).get("http_url", "http://localhost:8765")

    @property
    def oppai_oracle_provider(self) -> str | None:
        return self.data.get("oppai_oracle", {}).get("provider")

    @property
    def wd14(self) -> dict[str, Any]:
        return self.data.get("wd14", {})

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.data.get("sd_scripts_dir"):
            errors.append("sd_scripts_dir が未設定です (sd-scripts チェックアウトの絶対パス)")
        else:
            d = self.sd_scripts_dir
            if not d.is_dir():
                errors.append(f"sd_scripts_dir が存在しません: {d}")
            elif not (d / "sdxl_train_network.py").exists():
                errors.append(f"{d} は sd-scripts のディレクトリではないようです (sdxl_train_network.py が見つかりません)")
        backend = self.default_backend
        if backend not in KNOWN_BACKENDS:
            errors.append(f"defaults.backend が不正です: {backend!r} (有効値: {', '.join(KNOWN_BACKENDS)})")
        preset = self.default_preset
        if preset not in KNOWN_PRESETS:
            errors.append(f"defaults.preset が不正です: {preset!r} (有効値: {', '.join(KNOWN_PRESETS)})")
        return errors


# ---------------------------------------------------------------------------
# Character config
# ---------------------------------------------------------------------------


@dataclass
class CharacterConfig:
    data: dict[str, Any]
    name: str
    path: Path | None = None

    @classmethod
    def load(cls, name: str, characters_dir: Path | None = None) -> "CharacterConfig":
        base = characters_dir or (Path.cwd() / "characters")
        path = base / name / "character.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"キャラ設定が見つかりません: {path}\n"
                f"先に `lorayaki new {name}` を実行してください。"
            )
        data = deep_merge(CHARACTER_DEFAULTS, _load_yaml(path))
        if not data.get("name"):
            data["name"] = name
        return cls(data=data, name=name, path=path)

    def save(self) -> None:
        if self.path is None:
            self.path = paths.character_config_path(self.name)
        _dump_yaml(self.path, self.data)

    # -- typed accessors ----------------------------------------------------

    @property
    def trigger(self) -> str:
        return (self.data.get("trigger") or "").strip()

    @property
    def backend(self, ) -> str | None:
        return self.data.get("backend")

    @property
    def base_model(self) -> str | None:
        return self.data.get("base_model")

    @property
    def network(self) -> dict[str, Any]:
        return self.data.get("network", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.data.get("training", {})

    @property
    def tagging(self) -> dict[str, Any]:
        return self.data.get("tagging", {})

    @property
    def samples(self) -> dict[str, Any]:
        return self.data.get("samples", {})

    @property
    def caption_extension(self) -> str:
        return self.tagging.get("caption_extension", ".txt")

    @property
    def always_first_tags(self) -> list[str]:
        return list(self.tagging.get("always_first_tags") or [])

    @property
    def keep_tokens(self) -> int:
        """Number of leading comma-separated tokens sd-scripts must keep fixed:
        the trigger (1 token) plus any always_first_tags."""
        return 1 + len(self.always_first_tags)

    def resolve_backend(self, global_config: GlobalConfig) -> str:
        return self.backend or global_config.default_backend

    def resolve_preset(self, global_config: GlobalConfig) -> str:
        return self.training.get("preset") or global_config.default_preset

    def validate(self, global_config: GlobalConfig | None = None) -> list[str]:
        errors: list[str] = []
        if not self.trigger:
            errors.append("trigger が未設定です (例: 'cyk girl')")
        if not self.base_model:
            errors.append("base_model が未設定です (グローバル設定の models キーを指定)")
        elif global_config is not None and self.base_model not in global_config.models:
            errors.append(
                f"base_model '{self.base_model}' がグローバル設定の models に登録されていません "
                f"(登録済み: {', '.join(global_config.models) or 'なし'})"
            )
        backend = self.backend
        if backend is not None and backend not in KNOWN_BACKENDS:
            errors.append(f"backend が不正です: {backend!r} (有効値: {', '.join(KNOWN_BACKENDS)})")
        preset = self.training.get("preset")
        if preset is not None and preset not in KNOWN_PRESETS:
            errors.append(f"training.preset が不正です: {preset!r} (有効値: {', '.join(KNOWN_PRESETS)})")
        net = self.network
        for pair in (("dim", "alpha"), ("conv_dim", "conv_alpha")):
            if net.get(pair[0]) is not None and net.get(pair[1]) is None:
                errors.append(f"network.{pair[1]} も設定してください (network.{pair[0]} が設定されています)")
        return errors


def character_template() -> dict[str, Any]:
    """The character.yaml scaffold written by `lorayaki new`."""
    data = copy.deepcopy(CHARACTER_DEFAULTS)
    # Present only the fields a user typically edits; the rest is merged in
    # from CHARACTER_DEFAULTS at load time.
    return {
        "name": data["name"],
        "trigger": None,
        "backend": None,
        "base_model": None,
        "network": data["network"],
        "training": {
            "epochs": data["training"]["epochs"],
            "num_repeats": data["training"]["num_repeats"],
            "preset": data["training"]["preset"],
        },
        "tagging": {
            "drop_categories": data["tagging"]["drop_categories"],
            "extra_remove_tags": data["tagging"]["extra_remove_tags"],
            "always_first_tags": data["tagging"]["always_first_tags"],
        },
        "samples": {
            "prompts": data["samples"]["prompts"],
        },
    }
