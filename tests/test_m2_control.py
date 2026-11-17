import unittest

import numpy as np

from possession.m2_control import channels, confirmations, expand, ownership, stillness

CONFIG = {"stride": 5, "window": 5, "keeper_half_exclusion": False,
          "keeper_distance": 3., "control_distance": 2., "relative_speed": 1.,
          "control_speed": 5., "team_margin": .5, "restart_distance": 10.,
          "restart_speed": .5, "confirmation": 3, "restart_window": 3, "restart_votes": 1}


def features(n=10, distances=((1., 5.),), speed=2., relative=0., tagged=None,
             keepers=(False, False), half_ok=True):
    slots = len(distances[0])
    return {
        "distance": np.tile(np.asarray(distances, float), (n, 1)),
        "relative_speed": np.full((n, slots, 3), relative),
        "ball_speed": np.full((n, 3), speed),
        "half_ok": np.full((n, slots), half_ok),
        "teams": np.arange(slots) % 2,
        "keepers": np.asarray(keepers, bool),
        "tagged": np.zeros(n, bool) if tagged is None else np.asarray(tagged, bool),
        "n_frames": np.asarray(n * 5),
    }


class ChannelsTests(unittest.TestCase):
    def test_closer_team_within_control_distance_owns_the_ball(self):
        control, restart = channels(features(), CONFIG)
        np.testing.assert_array_equal(control, np.zeros(10, np.int8))
        np.testing.assert_array_equal(restart, np.zeros(10, np.int8))

    def test_team_margin_blocks_control_when_distances_are_close(self):
        control, _ = channels(features(distances=((1., 1.4),)), CONFIG)
        np.testing.assert_array_equal(control, np.full(10, -1, np.int8))

    def test_ball_outside_control_distance_is_uncontrolled(self):
        control, _ = channels(features(distances=((4., 5.),)), CONFIG)
        np.testing.assert_array_equal(control, np.full(10, -1, np.int8))

    def test_keeper_radius_applies_to_the_keeper_slot(self):
        tight = dict(CONFIG, control_distance=1.5)
        control, _ = channels(features(distances=((2.5, 5.),), keepers=(True, False)), tight)
        np.testing.assert_array_equal(control, np.zeros(10, np.int8))
        no_keeper, _ = channels(features(distances=((2.5, 5.),)), tight)
        np.testing.assert_array_equal(no_keeper, np.full(10, -1, np.int8))

    def test_keeper_half_exclusion_removes_a_keeper_out_of_position(self):
        config = dict(CONFIG, keeper_half_exclusion=True)
        control, _ = channels(features(keepers=(True, False), half_ok=False), config)
        np.testing.assert_array_equal(control, np.full(10, -1, np.int8))

    def test_restart_needs_ball_speed_above_the_restart_speed(self):
        _, slow = channels(features(speed=.4), CONFIG)
        np.testing.assert_array_equal(slow, np.full(10, -1, np.int8))


class ConfirmationTests(unittest.TestCase):
    def test_confirmation_needs_a_full_window_of_the_same_team(self):
        control = np.array([0, 0, 1, 0, 0], np.int8)
        restart = np.zeros(5, np.int8)
        confirmed, _ = confirmations(control, restart, dict(CONFIG, confirmation=3))
        np.testing.assert_array_equal(confirmed, np.array([-1, -1, -1, -1, -1], np.int8))

    def test_confirmation_appears_after_the_window_fills(self):
        control = np.zeros(5, np.int8)
        confirmed, voted = confirmations(control, np.zeros(5, np.int8), CONFIG)
        np.testing.assert_array_equal(confirmed, np.array([-1, -1, 0, 0, 0], np.int8))
        np.testing.assert_array_equal(voted, np.zeros(5, np.int8))

    def test_restart_votes_are_counted_within_the_window(self):
        restart = np.array([0, -1, -1, 0, 0], np.int8)
        _, voted = confirmations(np.full(5, -1, np.int8), restart,
                                 dict(CONFIG, restart_votes=2))
        np.testing.assert_array_equal(voted, np.array([-1, -1, -1, -1, 0], np.int8))


class OwnershipTests(unittest.TestCase):
    def test_owner_stays_unknown_until_restart_votes_reach_the_threshold(self):
        config = dict(CONFIG, restart_votes=2)
        labels = ownership(features(), config)
        np.testing.assert_array_equal(labels, np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int8))

    def test_tag_start_resets_the_owner_to_unknown(self):
        tagged = np.zeros(10, bool)
        tagged[5] = True
        config = dict(CONFIG, restart_votes=2)
        labels = ownership(features(tagged=tagged), config)
        np.testing.assert_array_equal(labels, np.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0], np.int8))

    def test_forced_dead_frames_are_unknown_and_reset_the_run(self):
        forced = np.zeros(10, bool)
        forced[4] = True
        config = dict(CONFIG, restart_votes=2)
        labels = ownership(features(), config, forced)
        np.testing.assert_array_equal(labels, np.array([2, 0, 0, 0, 2, 2, 0, 0, 0, 0], np.int8))


class StillnessTests(unittest.TestCase):
    SETTINGS = {"speed": 2., "samples": 2}

    def test_quiet_ball_stays_dead_while_no_restart_vote_arrives(self):
        dead = stillness(features(speed=1.), dict(CONFIG, restart_votes=2), self.SETTINGS)
        self.assertTrue(dead.all())

    def test_moving_ball_is_alive(self):
        dead = stillness(features(speed=4.), CONFIG, self.SETTINGS)
        self.assertFalse(dead.any())

    def test_tag_start_returns_the_ball_to_dead(self):
        tagged = np.zeros(10, bool)
        tagged[5] = True
        dead = stillness(features(speed=4., tagged=tagged), CONFIG, self.SETTINGS)
        self.assertFalse(dead[:5].any())
        self.assertTrue(dead[5])


class ExpandTests(unittest.TestCase):
    def test_labels_repeat_per_frame_and_stop_at_n_frames(self):
        labels = np.array([0, 1, 2], np.int8)
        data = features(n=3)
        data["n_frames"] = np.asarray(12)
        np.testing.assert_array_equal(expand(labels, data, 5), np.array(list("HHHHHAAAAADD")))


if __name__ == "__main__":
    unittest.main()
