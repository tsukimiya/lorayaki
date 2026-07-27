"""Tag merge pipeline: ordering, dedup, trigger pinning."""

from __future__ import annotations

from lorayaki.taggers.merge import caption_to_tags, merge_tags, tags_to_caption


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


class TestCaptionRoundtrip:
    def test_roundtrip(self):
        tags = ["cyk girl", "1girl", "long hair"]
        assert caption_to_tags(tags_to_caption(tags)) == tags

    def test_strips_empty_segments(self):
        assert caption_to_tags("a, , b ,, c") == ["a", "b", "c"]
