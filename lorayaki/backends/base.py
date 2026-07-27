"""Backend protocol: each model family translates a CharacterConfig into a
concrete sd-scripts training command. Tagging / dataset / sample generation
are backend-agnostic and shared."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from lorayaki.config import CharacterConfig, GlobalConfig


@dataclass(frozen=True)
class PresetConfig:
    """VRAM-oriented training preset."""

    network_dim: int
    network_alpha: int
    conv_dim: int
    conv_alpha: int
    batch_size: int
    resolution: int
    min_bucket_reso: int
    max_bucket_reso: int
    bucket_reso_steps: int


@dataclass(frozen=True)
class TrainPaths:
    """Resolved filesystem inputs/outputs for one training run."""

    base_model: Path
    dataset_toml: Path
    sample_prompts: Path
    output_dir: Path
    output_name: str


@runtime_checkable
class Backend(Protocol):
    name: str
    presets: dict[str, PresetConfig]

    def validate_config(self, config: CharacterConfig, global_config: GlobalConfig) -> list[str]:
        """Backend-specific config validation; returns error strings."""
        ...

    def build_train_command(
        self,
        *,
        config: CharacterConfig,
        global_config: GlobalConfig,
        preset: PresetConfig,
        paths: TrainPaths,
        resume_dir: Path | None = None,
    ) -> list[str]:
        """Full command list, e.g. ['accelerate', 'launch', ..., 'sdxl_train_network.py', ...]."""
        ...
