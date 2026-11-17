"""W3 set-piece observations, canonical tags and metadata team mapping."""

import unittest

import numpy as np

from possession.tag_features import NAMES, bits, canonicalize_intervals, token


def row(code="setpieces", team="h", kind="corner", start=0, end=10):
    return {"code": code, "start_frame": start, "end_frame": end,
            "tags": {"team": team, "type": kind}}


class TokenTests(unittest.TestCase):
    def test_tokens_are_casefolded_alphanumerics(self):
        self.assertEqual(token("Throw-In"), "throwin")
        self.assertEqual(token("Goal kick"), "goalkick")
        self.assertEqual(token("corner_kick"), "cornerkick")

    def test_missing_values_tokenize_to_empty(self):
        self.assertEqual(token(None), "")


class CanonicalizeTests(unittest.TestCase):
    def test_tags_are_normalized_without_mutating_the_input(self):
        intervals = [row(team="h", kind="Throw-In")]
        tags = canonicalize_intervals(intervals)[0]["tags"]
        self.assertEqual((tags["Team"], tags["Type"]), ("H", "throw_in"))
        self.assertEqual(intervals[0]["tags"], {"team": "h", "type": "Throw-In"})

    def test_unknown_team_and_type_become_none(self):
        tags = canonicalize_intervals([row(team="Someone", kind="Something")])[0]["tags"]
        self.assertEqual((tags["Team"], tags["Type"]), (None, None))

    def test_metadata_order_maps_the_two_names(self):
        tags = canonicalize_intervals([row(team="Reds")], ["Reds", "Blues"])[0]["tags"]
        self.assertEqual(tags["Team"], "H")
        tags = canonicalize_intervals([row(team="Blues")], ["Reds", "Blues"])[0]["tags"]
        self.assertEqual(tags["Team"], "A")


class BitsTests(unittest.TestCase):
    def test_observations_land_on_five_frame_samples(self):
        values = bits([row(start=0, end=12)], 20)
        self.assertEqual(values.shape, (4, len(NAMES)))
        np.testing.assert_array_equal(values[:, NAMES.index("setpiece_home")], [1, 1, 1, 0])
        np.testing.assert_array_equal(values[:, NAMES.index("setpiece_corner")], [1, 1, 1, 0])

    def test_active_window_is_clipped_to_the_track(self):
        values = bits([row(start=-10, end=5)], 10)
        np.testing.assert_array_equal(values[:, 0], [1, 0])

    def test_away_team_sets_the_away_column(self):
        values = bits([row(team="a")], 5)
        np.testing.assert_array_equal(values[0, :2], [0, 1])

    def test_only_setpieces_rows_contribute(self):
        values = bits([row(code="goals")], 10)
        np.testing.assert_array_equal(values, np.zeros((2, len(NAMES))))

    def test_unknown_type_keeps_the_team_bit(self):
        values = bits([row(kind="unknown")], 5)
        self.assertEqual(values[0, 0], 1)
        self.assertEqual(values[0, 2:].sum(), 0)

    def test_unknown_team_keeps_the_type_bit(self):
        values = bits([row(team="someone")], 5)
        self.assertEqual(values[0, 0], 0)
        self.assertEqual(values[0, NAMES.index("setpiece_corner")], 1)

    def test_stride_and_negative_frame_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "^W3 tag observations require the fixed five-frame stride$"):
            bits([], 10, stride=10)
        with self.assertRaisesRegex(ValueError, "^W3 tag observations require the fixed five-frame stride$"):
            bits([], -1)

    def test_empty_intervals_are_all_zero(self):
        np.testing.assert_array_equal(bits([], 12), np.zeros((3, len(NAMES))))


if __name__ == "__main__":
    unittest.main()
