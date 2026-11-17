"""Fixed m2 feature bank: shape, keeper half, absence and tag windows."""
import unittest

import numpy as np

from possession.data.tracking import Tracking
from possession.features.observations import BASE_STRIDE, WINDOWS, extract


def tracking(frames=20, ball_x=1.0, keeper_away_x=80.0):
    """Two home and two away slots, both teams keeping a keeper."""
    ball = np.column_stack([np.full(frames, ball_x), np.zeros(frames)])
    players = np.zeros((frames, 4, 2))
    players[:, 0, 0] = 10.0
    players[:, 1, 0] = 30.0
    players[:, 2, 0] = keeper_away_x
    players[:, 3, 0] = 70.0
    teams = np.array([0, 0, 1, 1], dtype=np.int8)
    keepers = np.array([True, False, True, False])
    return Tracking(players, ball, teams, keepers, True)


def tag(code, start, end):
    return {"code": code, "start_frame": start, "end_frame": end}


class ExtractTests(unittest.TestCase):
    def test_reports_every_documented_column(self):
        features = extract(tracking(), [])
        names = list(features["feature_names"])
        self.assertEqual(len(names), features["dead_features"].shape[1])
        self.assertEqual(names[:3], [f"ball_speed_{w}" for w in WINDOWS])
        self.assertEqual(names[-4:], ["since_tag_start", "until_tag_start",
                                      "since_tag_end", "until_tag_end"])
        self.assertEqual(features["dead_features"].shape, (20 // BASE_STRIDE, len(names)))
        self.assertEqual(int(features["n_frames"]), 20)
        self.assertEqual(features["teams"].tolist(), [0, 0, 1, 1])

    def test_keepers_must_stand_in_the_half_they_defend(self):
        home = extract(tracking(), [])
        self.assertTrue(home["half_ok"].all())

        crossed = extract(tracking(keeper_away_x=10.0), [])
        self.assertTrue(crossed["half_ok"][:, 0].all())
        self.assertFalse(crossed["half_ok"][:, 2].any())
        self.assertTrue(crossed["half_ok"][:, 1].all())

    def test_never_seen_ball_reports_missing_absence(self):
        features = extract(tracking(ball_x=np.nan), [])
        column = list(features["feature_names"]).index("last_off_pitch_m")
        self.assertTrue(np.isnan(features["dead_features"][:, column]).all())

    def test_tags_fill_their_span_and_time_their_edges(self):
        features = extract(tracking(), [tag("SET PIECES", 0, 10),
                                        tag("GOALS", 10, 20),
                                        tag("OTHER", 0, 20)])
        names = list(features["feature_names"])
        dead = features["dead_features"]
        self.assertEqual(dead[:, names.index("set_pieces")].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(dead[:, names.index("goals")].tolist(), [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(features["tagged"].tolist(), [True, True, True, True])
        np.testing.assert_allclose(dead[:, names.index("since_tag_start")],
                                   [np.nan, 0.2, 0.4, 0.2], rtol=0, atol=1e-6)
        np.testing.assert_allclose(dead[:, names.index("until_tag_start")],
                                   [0.0, 0.2, 0.0, np.nan], rtol=0, atol=1e-6)

    def test_empty_or_inverted_intervals_leave_no_tag_evidence(self):
        features = extract(tracking(), [tag("SET PIECES", 12, 12)])
        dead = features["dead_features"]
        self.assertFalse(features["tagged"].any())
        self.assertTrue(np.isnan(dead[:, list(features["feature_names"])
                                       .index("since_tag_start")]).all())
