"""Golden test: the Illustrious training command must match the verified
reference (plan Appendix), and must never pass --text_encoder_lr in
unet-only mode."""

from __future__ import annotations

import shlex
import tomli
from pathlib import Path

from lorayaki.backends import get_backend
from lorayaki.backends.base import TrainPaths
from lorayaki.backends.illustrious import PRESETS
from lorayaki.config import CharacterConfig, GlobalConfig
from lorayaki.cli import main


def _expected_command(model: Path, toml: Path, prompts: Path, outdir: Path) -> list[str]:
    return [
        "accelerate", "launch", "--num_cpu_threads_per_process", "1",
        "sdxl_train_network.py",
        "--pretrained_model_name_or_path", str(model),
        "--dataset_config", str(toml),
        "--network_module", "networks.lora",
        "--network_dim", "32",
        "--network_alpha", "16",
        "--network_args", "conv_dim=16", "conv_alpha=8",
        "--learning_rate", "0.0001",
        "--unet_lr", "0.0001",
        "--optimizer_type", "AdamW8bit",
        "--lr_scheduler", "cosine_with_restarts",
        "--lr_scheduler_num_cycles", "3",
        "--max_train_epochs", "10",
        "--save_every_n_epochs", "1",
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--network_train_unet_only",
        "--xformers",
        "--max_grad_norm", "1",
        "--resolution", "1024",
        "--enable_bucket",
        "--bucket_reso_steps", "64",
        "--min_bucket_reso", "512",
        "--max_bucket_reso", "1536",
        "--output_dir", str(outdir),
        "--output_name", "testchar",
        "--save_model_as", "safetensors",
        "--sample_every_n_epochs", "1",
        "--sample_prompts", str(prompts),
        "--sample_sampler", "euler_a",
        "--noise_offset", "0.03",  # default training.extra_args
    ]


def test_build_train_command_golden(project: Path, sample_char: str):
    gcfg = GlobalConfig.load()
    ccfg = CharacterConfig.load(sample_char)
    backend = get_backend("illustrious")
    model = gcfg.models["illustrious-xl-0.1"]
    toml = project / "characters/testchar/work/dataset.toml"
    prompts = project / "characters/testchar/work/sample_prompts.txt"
    outdir = project / "characters/testchar/work/output"

    cmd = backend.build_train_command(
        config=ccfg,
        global_config=gcfg,
        preset=PRESETS["16gb"],
        paths=TrainPaths(model, toml, prompts, outdir, "testchar"),
    )
    assert cmd == _expected_command(model, toml, prompts, outdir)
    assert "--text_encoder_lr" not in cmd  # unet-only: never pass it


def test_config_overrides_preset(project: Path, sample_char: str):
    gcfg = GlobalConfig.load()
    ccfg = CharacterConfig.load(sample_char)
    ccfg.data["network"] = {"dim": 64, "alpha": 32, "conv_dim": 4, "conv_alpha": 2}
    backend = get_backend("illustrious")
    cmd = backend.build_train_command(
        config=ccfg,
        global_config=gcfg,
        preset=PRESETS["12gb"],
        paths=TrainPaths(gcfg.models["illustrious-xl-0.1"], Path("d.toml"), Path("s.txt"), Path("o"), "testchar"),
    )
    i = cmd.index("--network_dim")
    assert cmd[i : i + 6] == ["--network_dim", "64", "--network_alpha", "32",
                              "--network_args", "conv_dim=4"]
    assert "conv_alpha=2" in cmd


def test_extra_args_passthrough_and_resume(project: Path, sample_char: str):
    gcfg = GlobalConfig.load()
    ccfg = CharacterConfig.load(sample_char)
    ccfg.data["training"]["extra_args"] = {"blocks_to_swap": 4, "sdpa": True, "skip": False}
    backend = get_backend("illustrious")
    cmd = backend.build_train_command(
        config=ccfg,
        global_config=gcfg,
        preset=PRESETS["16gb"],
        paths=TrainPaths(gcfg.models["illustrious-xl-0.1"], Path("d.toml"), Path("s.txt"), Path("o"), "testchar"),
        resume_dir=Path("/state/last-state"),
    )
    assert cmd[cmd.index("--blocks_to_swap") + 1] == "4"
    assert "--sdpa" in cmd
    assert "--skip" not in cmd
    assert cmd[cmd.index("--resume") + 1] == str(Path("/state/last-state"))


def test_dry_run_end_to_end(project: Path, sample_char: str, capsys):
    # prep artifacts first (prep only needs config + images)
    assert main(["prep", "testchar"]) == 0
    dataset_config = tomli.loads(
        (project / "characters/testchar/work/dataset.toml").read_text(encoding="utf-8")
    )
    assert dataset_config["general"]["shuffle_caption"] is False
    rc = main(["train", "testchar", "--dry-run"])
    assert rc == 0
    printed = capsys.readouterr().out.strip()
    cmd = shlex.split(printed)
    assert cmd[0] == "accelerate"
    assert "sdxl_train_network.py" in cmd
    assert "--text_encoder_lr" not in cmd
    assert "--dataset_config" in cmd
    # preset 16gb defaults
    assert cmd[cmd.index("--network_dim") + 1] == "32"


def test_anima_stub_raises(project: Path, sample_char: str):
    gcfg = GlobalConfig.load()
    ccfg = CharacterConfig.load(sample_char)
    ccfg.data["backend"] = "anima"
    backend = get_backend("anima")
    errors = backend.validate_config(ccfg, gcfg)
    assert errors  # missing anima model keys / presets
    try:
        backend.build_train_command()
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass
