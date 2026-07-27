"""JoyCaption subprocess command assembly (pure; no subprocess executed)."""

from __future__ import annotations

from pathlib import Path

from lorayaki.taggers.joycaption import build_joycaption_command


def test_build_joycaption_command_defaults():
    cmd = build_joycaption_command(
        "/jc/venv/bin/python",
        images_file=Path("/tmp/images.txt"),
        model="fancyfeast/llama-joycaption-beta-one-hf-llava",
        prompt="Write a long descriptive caption for this image.",
    )
    assert cmd[0] == "/jc/venv/bin/python"
    assert cmd[1].endswith("_joycaption_runner.py")
    assert cmd[cmd.index("--images") + 1] == str(Path("/tmp/images.txt"))
    assert cmd[cmd.index("--model") + 1] == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    assert cmd[cmd.index("--max-new-tokens") + 1] == "256"
    assert cmd[cmd.index("--batch-size") + 1] == "1"
    assert "--greedy" not in cmd


def test_build_joycaption_command_custom():
    cmd = build_joycaption_command(
        "python",
        images_file=Path("imgs.txt"),
        model="custom/model",
        prompt="Describe.",
        batch_size=4,
        max_new_tokens=512,
        greedy=True,
        temperature=0.8,
        top_p=0.95,
    )
    assert cmd[cmd.index("--batch-size") + 1] == "4"
    assert cmd[cmd.index("--max-new-tokens") + 1] == "512"
    assert cmd[cmd.index("--temperature") + 1] == "0.8"
    assert cmd[cmd.index("--top-p") + 1] == "0.95"
    assert "--greedy" in cmd
