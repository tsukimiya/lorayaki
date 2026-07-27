"""Illustrious (SDXL) LoCon backend — sdxl_train_network.py.

Verified against sd-scripts: networks/lora.py enables Conv2d 3x3 LoRA targets
(ResnetBlock2D / Downsample2D / Upsample2D) when conv_dim is passed via
--network_args, which is what makes this LoCon. text_encoder_lr is
deliberately never passed: with cache_text_encoder_outputs +
network_train_unet_only the TE is frozen and the flag is irrelevant (and its
nargs="*" parsing is a footgun).
"""

from __future__ import annotations

from pathlib import Path

from lorayaki.config import CharacterConfig, GlobalConfig
from lorayaki.backends.base import PresetConfig, TrainPaths

PRESETS: dict[str, PresetConfig] = {
    "12gb": PresetConfig(16, 16, 8, 8, 1, 1024, 512, 1536, 64),
    "16gb": PresetConfig(32, 16, 16, 8, 2, 1024, 512, 1536, 64),
    "24gb": PresetConfig(32, 16, 16, 8, 4, 1024, 512, 2048, 64),
}


def _fmt(value: float | int) -> str:
    """Format numbers for CLI args (1e-4 -> '0.0001', 1.0 -> '1')."""
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


class IllustriousBackend:
    name = "illustrious"
    presets = PRESETS
    train_script = "sdxl_train_network.py"

    def resolve_network(self, config: CharacterConfig, preset: PresetConfig) -> tuple[int, int, int, int]:
        """(dim, alpha, conv_dim, conv_alpha); character.yaml overrides preset."""
        net = config.network
        dim = int(net.get("dim") or preset.network_dim)
        alpha = int(net.get("alpha") or preset.network_alpha)
        conv_dim = int(net.get("conv_dim") or preset.conv_dim)
        conv_alpha = int(net.get("conv_alpha") or preset.conv_alpha)
        return dim, alpha, conv_dim, conv_alpha

    def validate_config(self, config: CharacterConfig, global_config: GlobalConfig) -> list[str]:
        errors: list[str] = []
        t = config.training
        if t.get("cache_text_encoder_outputs") and not t.get("network_train_unet_only"):
            errors.append(
                "cache_text_encoder_outputs を使う場合は network_train_unet_only: true が必要です"
            )
        preset_name = config.resolve_preset(global_config)
        preset = PRESETS.get(preset_name)
        if preset is None:
            errors.append(f"不明なプリセット: {preset_name}")
        else:
            if preset.max_bucket_reso % preset.bucket_reso_steps != 0:
                errors.append("max_bucket_reso は bucket_reso_steps の倍数である必要があります")
        return errors

    def build_train_command(
        self,
        *,
        config: CharacterConfig,
        global_config: GlobalConfig,
        preset: PresetConfig,
        paths: TrainPaths,
        resume_dir: Path | None = None,
    ) -> list[str]:
        t = config.training
        dim, alpha, conv_dim, conv_alpha = self.resolve_network(config, preset)

        cmd: list[str] = [
            "accelerate", "launch", "--num_cpu_threads_per_process", "1",
            self.train_script,
            "--pretrained_model_name_or_path", str(paths.base_model),
            "--dataset_config", str(paths.dataset_toml),
            "--network_module", "networks.lora",
            "--network_dim", str(dim),
            "--network_alpha", str(alpha),
            "--network_args", f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
            "--learning_rate", _fmt(float(t.get("unet_lr", 1e-4))),
            "--unet_lr", _fmt(float(t.get("unet_lr", 1e-4))),
            "--optimizer_type", str(t.get("optimizer", "AdamW8bit")),
            "--lr_scheduler", str(t.get("scheduler", "cosine_with_restarts")),
        ]
        sched_args = t.get("scheduler_args") or {}
        if sched_args:
            cmd += ["--lr_scheduler_args", *(f"{k}={v}" for k, v in sched_args.items())]
        cmd += [
            "--max_train_epochs", str(int(t.get("epochs", 10))),
            "--save_every_n_epochs", str(int(t.get("save_every_n_epochs", 1))),
            "--mixed_precision", str(t.get("mixed_precision", "bf16")),
        ]
        for flag, key in (
            ("--gradient_checkpointing", "gradient_checkpointing"),
            ("--cache_latents", "cache_latents"),
            ("--cache_text_encoder_outputs", "cache_text_encoder_outputs"),
            ("--network_train_unet_only", "network_train_unet_only"),
            ("--xformers", "xformers"),
        ):
            if t.get(key):
                cmd.append(flag)
        cmd += [
            "--max_grad_norm", _fmt(float(t.get("max_grad_norm", 1.0))),
        ]
        if t.get("seed") is not None:
            cmd += ["--seed", str(int(t["seed"]))]
        cmd += [
            "--resolution", str(preset.resolution),
            "--enable_bucket",
            "--bucket_reso_steps", str(preset.bucket_reso_steps),
            "--min_bucket_reso", str(preset.min_bucket_reso),
            "--max_bucket_reso", str(preset.max_bucket_reso),
            "--output_dir", str(paths.output_dir),
            "--output_name", paths.output_name,
            "--save_model_as", "safetensors",
            "--sample_every_n_epochs", str(int(t.get("sample_every_n_epochs", 1))),
            "--sample_prompts", str(paths.sample_prompts),
            "--sample_sampler", str(config.samples.get("sampler", "euler_a")),
        ]
        if resume_dir is not None:
            cmd += ["--resume", str(resume_dir)]
        # Backend/user pass-through flags (--key value or --flag for bools)
        for key, value in (t.get("extra_args") or {}).items():
            flag = f"--{key}"
            if value is True:
                cmd.append(flag)
            elif value is False or value is None:
                continue
            else:
                cmd += [flag, str(value)]
        return cmd
