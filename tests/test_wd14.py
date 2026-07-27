"""WD14 subprocess command assembly (pure; no subprocess executed)."""

from __future__ import annotations

from pathlib import Path

from lorayaki.taggers.wd14 import build_wd14_command


def test_build_wd14_command_defaults():
    image_dir = Path("/data/characters/foo/images")
    cmd = build_wd14_command(
        "/sd/venv/bin/python",
        image_dir,
        repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
    )
    assert cmd[0] == "/sd/venv/bin/python"
    assert cmd[1] == str(Path("finetune") / "tag_images_by_wd14_tagger.py")
    assert str(image_dir) in cmd
    assert "--onnx" in cmd
    assert "--remove_underscore" in cmd
    assert "--character_tag_expand" in cmd
    assert cmd[cmd.index("--repo_id") + 1] == "SmilingWolf/wd-v1-4-convnext-tagger-v2"
    assert cmd[cmd.index("--general_threshold") + 1] == "0.35"
    assert cmd[cmd.index("--character_threshold") + 1] == "0.85"
    assert "--undesired_tags" not in cmd


def test_build_wd14_command_custom():
    cmd = build_wd14_command(
        "python",
        Path("imgs"),
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        onnx=False,
        batch_size=8,
        general_threshold=0.4,
        character_threshold=0.9,
        remove_underscore=False,
        character_tag_expand=False,
        undesired_tags=["watermark", "text"],
    )
    assert "--onnx" not in cmd
    assert "--remove_underscore" not in cmd
    assert cmd[cmd.index("--batch_size") + 1] == "8"
    assert cmd[cmd.index("--undesired_tags") + 1] == "watermark,text"
