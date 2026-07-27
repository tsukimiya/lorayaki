"""Path helpers. All path handling goes through pathlib so the tool works
identically on Windows (development) and Linux (GPU training machine).

Conventions:
- The CLI is meant to be run from the lorayaki project root.
- Relative paths in the global config resolve against the CWD (standard CLI
  semantics); absolute paths (or ``~/...``) are recommended for sd_scripts_dir
  and model files, which usually live outside the project.
- Per-character layout::

    characters/<name>/
      images/            # user drops raw images here; captions are written
                         # next to each image as <image>.txt (user-editable)
      character.yaml     # per-character settings
      work/              # generated; gitignored
        dataset/         # assembled images + captions used for training
        dataset.toml     # sd-scripts dataset config
        sample_prompts.txt
        output/          # checkpoints + sample/ images
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def resolve_path(raw: str | Path | None, base: Path | None = None) -> Path | None:
    """Expand ``~`` and make relative paths absolute against *base* (default CWD)."""
    if raw is None:
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = (base or Path.cwd()) / p
    return p


def project_root() -> Path:
    return Path.cwd()


def characters_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "characters"


def character_dir(name: str, root: Path | None = None) -> Path:
    return characters_dir(root) / name


def images_dir(name: str, root: Path | None = None) -> Path:
    return character_dir(name, root) / "images"


def character_config_path(name: str, root: Path | None = None) -> Path:
    return character_dir(name, root) / "character.yaml"


def work_dir(name: str, root: Path | None = None) -> Path:
    return character_dir(name, root) / "work"


def dataset_dir(name: str, root: Path | None = None) -> Path:
    return work_dir(name, root) / "dataset"


def dataset_toml_path(name: str, root: Path | None = None) -> Path:
    return work_dir(name, root) / "dataset.toml"


def sample_prompts_path(name: str, root: Path | None = None) -> Path:
    return work_dir(name, root) / "sample_prompts.txt"


def output_dir(name: str, root: Path | None = None) -> Path:
    return work_dir(name, root) / "output"


def caption_path_for(image: Path, extension: str = ".txt") -> Path:
    """Caption file that belongs to *image* (same stem, given extension)."""
    return image.with_suffix(extension)


def list_images(directory: Path) -> list[Path]:
    """All images directly under *directory*, sorted by name for determinism."""
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower(),
    )
