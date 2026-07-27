"""Anima backend (P1atDev Anima, DiT + Qwen3 TE + Qwen-Image VAE).

Stub: config validation is implemented so character.yaml files can be
prepared today; build_train_command raises until Anima support is built.

Known differences from Illustrious (from sd-scripts docs/anima_train_network.md):
- train script: anima_train_network.py
- network module: networks.lora_anima (targets self/cross attn + MLP)
- extra required args: --qwen3 <path>, --vae <path>
- recommended: --timestep_sampling sigmoid, --qwen_image_vae_2d,
  --cache_text_encoder_outputs, --blocks_to_swap N (VRAM)
- bucket resolutions divisible by 16 (not 64)
Model registry keys: anima-dit, anima-qwen3, anima-vae.
"""

from __future__ import annotations

from lorayaki.config import CharacterConfig, GlobalConfig
from lorayaki.backends.base import PresetConfig, TrainPaths

# Defined properly when Anima support lands.
PRESETS: dict[str, PresetConfig] = {}

ANIMA_MODEL_KEYS = ("anima-dit", "anima-qwen3", "anima-vae")


class AnimaBackend:
    name = "anima"
    presets = PRESETS
    train_script = "anima_train_network.py"

    def validate_config(self, config: CharacterConfig, global_config: GlobalConfig) -> list[str]:
        errors: list[str] = []
        models = global_config.models
        if config.base_model:
            # Anima characters register the DiT as base_model; qwen3/vae come
            # from the registry too.
            for key in ("anima-qwen3", "anima-vae"):
                if key not in models:
                    errors.append(f"グローバル設定の models に '{key}' が必要です (Anima)")
        preset_name = config.resolve_preset(global_config)
        preset = PRESETS.get(preset_name)
        if preset is None:
            errors.append(
                f"Anima 用プリセット '{preset_name}' は未定義です "
                f"(Anima サポートは準備中 — training.extra_args で個別指定してください)"
            )
        elif preset.bucket_reso_steps % 16 != 0:
            errors.append("Anima では bucket_reso_steps は 16 の倍数である必要があります")
        return errors

    def build_train_command(self, **kwargs) -> list[str]:
        raise NotImplementedError(
            "Anima バックエンドの学習コマンド生成はまだ実装されていません。"
            "当面は Illustrious 系をご利用ください。"
        )
