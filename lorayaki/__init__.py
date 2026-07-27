"""lorayaki — quick character-LoRA pipeline for kohya sd-scripts.

Pipeline: images -> tagging (OppaiOracle + WD14) -> dataset assembly
-> LoCon training (Illustrious/SDXL now, Anima later) -> sample images.
"""

__version__ = "0.1.0"
