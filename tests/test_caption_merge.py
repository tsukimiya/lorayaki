"""Tag merge pipeline: ordering, dedup, trigger pinning."""

from __future__ import annotations

import numpy as np

from lorayaki.taggers.merge import caption_to_tags, merge_tags, tags_to_caption
from lorayaki.taggers.oppai_oracle import probs_to_tags


class TestMergeTags:
    def test_trigger_comes_first(self):
        out = merge_tags(["smile"], ["1girl"], trigger="cyk girl")
        assert out[0] == "cyk girl"

    def test_order_wd14_before_oppai_extras(self):
        out = merge_tags(
            oppai_tags=["huge breasts", "1girl"],   # 1girl is a dup
            wd14_tags=["1girl", "long hair"],
            trigger="cyk girl",
        )
        assert out == ["cyk girl", "1girl", "long hair", "huge breasts"]

    def test_confidence_order_preserved(self):
        out = merge_tags(None, ["b", "a", "c"], trigger="t")
        assert out == ["t", "b", "a", "c"]  # NOT alphabetical

    def test_dedup_case_insensitive_first_spelling_wins(self):
        out = merge_tags(["Long Hair"], ["long hair"], trigger="t")
        assert out == ["t", "long hair"]

    def test_whitespace_normalized(self):
        out = merge_tags(["solo  "], ["  solo"], trigger="t")
        assert out == ["t", "solo"]

    def test_always_first_after_trigger(self):
        out = merge_tags(["x"], ["y"], trigger="t", always_first_tags=["1girl", "solo"])
        assert out[:3] == ["t", "1girl", "solo"]

    def test_blacklist_applies_to_taggers_not_pinned(self):
        out = merge_tags(
            ["watermark", "smile"],
            ["jpeg artifacts", "1girl"],
            trigger="cyk girl",
            always_first_tags=["1girl"],
            extra_remove_tags=["watermark", "jpeg artifacts"],
        )
        assert out == ["cyk girl", "1girl", "smile"]

    def test_empty_inputs(self):
        assert merge_tags(None, None, trigger="t") == ["t"]
        assert merge_tags([], [], trigger="") == []

    def test_deterministic(self):
        a = merge_tags(["x", "y"], ["z"], trigger="t")
        b = merge_tags(["x", "y"], ["z"], trigger="t")
        assert a == b

    def test_oppai_underscore_tags_dedup_against_wd14(self):
        """End-to-end: OppaiOracle's danbooru-style tags are space-separated by
        probs_to_tags, so they dedup against WD14's space-separated output."""
        names = ["<PAD>", "polka_dot_bow", "solo"]
        cats = [0, 0, 0]
        probs = np.array([0.0, 0.99, 0.99], dtype=np.float32)
        oppai = probs_to_tags(probs, names, cats)  # ["polka dot bow", "solo"]
        merged = merge_tags(oppai, ["solo", "1girl"], trigger="t")
        assert merged == ["t", "solo", "1girl", "polka dot bow"]


class TestCaptionRoundtrip:
    def test_roundtrip(self):
        tags = ["cyk girl", "1girl", "long hair"]
        assert caption_to_tags(tags_to_caption(tags)) == tags

    def test_strips_empty_segments(self):
        assert caption_to_tags("a, , b ,, c") == ["a", "b", "c"]


class TestTagsToCaptionDescription:
    def test_no_description_unchanged(self):
        assert tags_to_caption(["a", "b"]) == "a, b"

    def test_description_appended_verbatim(self):
        tags = ["cyk girl", "1girl", "long hair"]
        desc = "A young woman with long flowing hair stands in a garden."
        result = tags_to_caption(tags, description=desc)
        assert result == (
            "cyk girl, 1girl, long hair, "
            "A young woman with long flowing hair stands in a garden."
        )

    def test_description_internal_commas_preserved(self):
        result = tags_to_caption(
            ["trigger"], description="She has blue eyes, long hair, and a smile."
        )
        assert result == "trigger, She has blue eyes, long hair, and a smile."

    def test_empty_tags_with_description(self):
        assert tags_to_caption([], description="Just a description.") == "Just a description."
