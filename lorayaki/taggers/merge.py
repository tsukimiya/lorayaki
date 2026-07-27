"""Tag merging: OppaiOracle + WD14 outputs -> one ordered caption per image.

Ordering policy (deterministic):
  1. trigger (single comma segment, may contain spaces, e.g. "cyk girl")
  2. always_first_tags (user-pinned)
  3. WD14 tags, in the tagger's confidence order
  4. OppaiOracle-only tags, in descending probability

Dedup is case-insensitive; the first occurrence's spelling wins. The trigger
and always_first_tags bypass the blacklist (explicit user intent).
"""

from __future__ import annotations

from typing import Iterable

DEFAULT_SEPARATOR = ", "


def _key(tag: str) -> str:
    return " ".join(tag.split()).casefold()


def merge_tags(
    oppai_tags: Iterable[str] | None,
    wd14_tags: Iterable[str] | None,
    *,
    trigger: str,
    always_first_tags: Iterable[str] | None = None,
    extra_remove_tags: Iterable[str] | None = None,
) -> list[str]:
    """Merge tag lists into one ordered, deduplicated caption segment list."""
    remove = {_key(t) for t in (extra_remove_tags or []) if t.strip()}
    out: list[str] = []
    seen: set[str] = set()

    def add(tag: str, *, apply_blacklist: bool = True) -> None:
        tag = " ".join(str(tag).split())
        if not tag:
            return
        key = _key(tag)
        if key in seen:
            return
        if apply_blacklist and key in remove:
            return
        seen.add(key)
        out.append(tag)

    # 1-2. pinned: trigger + always_first (blacklist-exempt)
    if trigger:
        add(trigger, apply_blacklist=False)
    for tag in always_first_tags or []:
        add(tag, apply_blacklist=False)

    # 3. WD14 (confidence order preserved)
    for tag in wd14_tags or []:
        add(tag)

    # 4. OppaiOracle extras (probability order preserved)
    for tag in oppai_tags or []:
        add(tag)

    return out


def tags_to_caption(
    tags: Iterable[str],
    *,
    separator: str = DEFAULT_SEPARATOR,
    description: str | None = None,
) -> str:
    """Join *tags* with *separator*, then append a verbatim natural-language
    *description* (e.g. JoyCaption output) after the tags. The description is
    kept as-is — its internal commas are preserved and it is never treated as
    a tag (never counted, never split on re-read)."""
    caption = separator.join(tags)
    if description:
        caption = f"{caption}{separator}{description}" if caption else description
    return caption


def caption_to_tags(caption: str, *, separator: str = DEFAULT_SEPARATOR) -> list[str]:
    # Split on the comma itself (not ", ") so malformed separators like
    # "a, , b" still round-trip cleanly.
    return [t.strip() for t in caption.strip().split(",") if t.strip()]
