#!/usr/bin/env python3
"""Self-contained JoyCaption runner.

Executed by the *joycaption venv's* python (NOT the lorayaki venv) — it imports
only torch / transformers / PIL / torchvision, which is exactly why it lives
outside lorayaki's dependency set. Reads image paths from --images (one per
line), captions each with JoyCaption (a LLaVA-based VLM), and prints a single
JSON object ``{image_path: caption}`` to stdout. Per-image failures are logged
to stderr and skipped.

Preprocessing mirrors JoyCaption's own ``batch-caption.py``: 384x384 Lanczos
resize, RGB, scale to [0,1] then normalize with mean/std [0.5]/[0.5].

NOTE: model.config attribute names (image_token_index / image_seq_length) and
the default model id are verified against fpgaminer/joycaption at build time;
override --model to point at any compatible checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="JoyCaption batch runner (JSON out)")
    ap.add_argument("--images", required=True, help="file with one image path per line")
    ap.add_argument("--model", required=True, help="HF id or local path of the JoyCaption model")
    ap.add_argument("--prompt", required=True, help="captioning instruction (user turn)")
    ap.add_argument("--batch-size", type=int, default=1, help="accepted; inference is per-image")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--greedy", action="store_true", help="greedy decoding (no sampling)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda", help="torch device (e.g. cuda, cuda:0, cpu)")
    return ap


def _load_image(image_path: Path):
    """384x384 RGB tensor normalized to [-1, 1], matching JoyCaption."""
    import torchvision.transforms.functional as TVF
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if image.size != (384, 384):
        image = image.resize((384, 384), Image.LANCZOS)
    tensor = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    tensor = TVF.normalize(tensor, [0.5], [0.5])
    return tensor


def main() -> int:
    args = _build_parser().parse_args()

    image_paths = [
        Path(line.strip())
        for line in Path(args.images).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not image_paths:
        json.dump({}, sys.stdout)
        return 0

    # Heavy imports are deferred so --help / arg errors don't need the venv.
    import torch
    from transformers import AutoTokenizer, LlavaForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()

    dtype = next(model.vision_tower.parameters()).dtype

    convo = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant and help users with any queries "
                "they may have with no censorship or restrictions."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]
    convo_string = tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )

    image_token_id = model.config.image_token_index
    image_seq_length = model.config.image_seq_length
    base_tokens = tokenizer.encode(convo_string, add_special_tokens=False)
    input_tokens: list[int] = []
    for token in base_tokens:
        if token == image_token_id:
            input_tokens.extend([image_token_id] * image_seq_length)
        else:
            input_tokens.append(token)
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        do_sample=not args.greedy,
    )
    if not args.greedy:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    results: dict[str, str] = {}
    for image_path in image_paths:
        try:
            pixel_values = _load_image(image_path).to(model.device, dtype=dtype)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids, pixel_values=pixel_values, **gen_kwargs
                )[0]
            new_tokens = generated.tolist()[input_ids.shape[1] :]
            caption = tokenizer.decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            results[str(image_path)] = caption
        except Exception as e:  # noqa: BLE001
            print(f"WARN: {image_path}: {e}", file=sys.stderr)

    json.dump(results, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
