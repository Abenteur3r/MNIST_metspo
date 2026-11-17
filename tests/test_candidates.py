"""Candidate dispatch: each family sees the same tracking boundary."""

import unittest
from unittest import mock

import numpy as np

from possession import candidates
from possession.io_atd import Tracking


def tracking(ball=((0., 0.),), players=((0., 0.), (5., 0.)), teams=(0, 1)):
    ball = np.asarray(ball, dtype=float).reshape(-1, 2)
    players = np.broadcast_to(np.asarray(players, dtype=float), (len(ball), len(teams), 2)).copy()
    return Tracking(players, ball, np.asarray(teams, dtype=np.int8), np.zeros(len(teams), dtype=bool), True)


class NearestTests(unittest.TestCase):
    def test_reports_the_closest_slot_team_and_distance(self):
        teams, distance, seen = candidates.nearest(tracking(players=((3., 4.), (1., 0.))))
        np.testing.assert_array_equal(teams, [1])
        np.testing.assert_allclose(distance, [1.])
        np.testing.assert_array_equal(seen, [True])

    def test_a_missing_ball_has_no_owner(self):
        teams, distance, seen = candidates.nearest(tracking(ball=(np.nan, 0.)))
        np.testing.assert_array_equal(teams, [-1])
        np.testing.assert_array_equal(seen, [False])


class PredictTests(unittest.TestCase):
    def test_rejects_an_unknown_candidate(self):
        with self.assertRaisesRegex(ValueError, "^Unknown candidate$"):
            candidates.predict("nope", tracking(), [], [(0, 0)])

    def test_solution_delegates_to_the_packaged_serving_path(self):
        from possession import shared_temporal_serving
        with mock.patch.object(shared_temporal_serving, "predict", return_value="sentinel") as shared:
            self.assertEqual(candidates.predict("solution", tracking(), [], [(0, 0)], team_names=[0, 1]),
                             "sentinel")
        shared.assert_called_once()
        self.assertEqual(shared.call_args.kwargs, {"team_names": [0, 1]})

    def test_control_v1_combines_evidence_dead_regions_and_the_decoder(self):
        with mock.patch.object(candidates, "evidence", return_value=("control", "restart")) as evidence, \
                mock.patch.object(candidates, "dead_region_mask", return_value="dead") as mask, \
                mock.patch.object(candidates, "decode", return_value="labels") as decode:
            self.assertEqual(candidates.predict("control_v1", tracking(), [], [(0, 0)]), "labels")
        evidence.assert_called_once()
        mask.assert_called_once_with(1, [])
        decode.assert_called_once_with("control", "restart", "dead", [(0, 0)])

    def test_baseline_keeps_each_block_independent(self):
        with mock.patch.object(candidates, "smooth", side_effect=lambda teams, width: ["H"] * len(teams)):
            labels = candidates.predict("baseline_smooth50", tracking(ball=((0., 0.),) * 4), [], [(1, 1)])
        np.testing.assert_array_equal(labels, ["X", "H", "X", "X"])

    def test_v7_maps_states_inside_blocks_only(self):
        states = np.array([0, 1, 2, 0])
        with mock.patch.object(candidates, "ball_velocity", return_value=("velocity", "valid")) as velocity, \
                mock.patch("possession.v7.v7_labels", return_value=states) as v7:
            labels = candidates.predict("v7", tracking(ball=((0., 0.),) * 4), [], [(0, 1)])
        np.testing.assert_array_equal(labels, ["D", "H", "X", "X"])
        velocity.assert_called_once()
        self.assertEqual(v7.call_args.kwargs, {"velocity_valid": "valid"})


if __name__ == "__main__":
    unittest.main()
