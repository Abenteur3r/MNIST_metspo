"""Solution command-line integration using invented metadata and mocked inference."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.candidates import ALL_CANDIDATES, predict
from possession import player_attention_serving, shared_temporal_serving
from possession.io_atd import Tracking
from possession.run import main
from possession.tag_features import canonicalize_intervals


class SolutionPublicationTests(unittest.TestCase):
    def test_candidate_dispatch_forwards_explicit_and_default_team_names(self):
        self.assertIn("solution", ALL_CANDIDATES)
        tracks, intervals, blocks = object(), [], [(0, 5)]
        for names in (None, ("Home Example", "Away Example")):
            with self.subTest(team_names=names), patch(
                    "possession.shared_temporal_serving.predict", return_value="sentinel") as call:
                kwargs = {} if names is None else {"team_names": names}
                self.assertEqual(predict("solution", tracks, intervals, blocks, **kwargs),
                                 "sentinel")
                call.assert_called_once_with(tracks, intervals, blocks, team_names=names)

    def test_predict_uses_the_packaged_mnist_members(self):
        tracking = Tracking(np.zeros((10, 8, 2)), np.zeros((10, 2)),
                            np.array([0]*4 + [1]*4), np.zeros(8, dtype=bool), True)
        with patch("possession.player_attention_serving.load_model",
                   side_effect=AssertionError("player-attention loader")):
            labels = shared_temporal_serving.predict(tracking, [], [(0, 9)])
        self.assertEqual(labels.shape, (10,))
        self.assertTrue(np.isin(labels, list("HAD")).all())

    def test_both_candidates_reject_invalid_blocks_and_intervals(self):
        tracking = Tracking(np.zeros((10, 8, 2)), np.zeros((10, 2)),
                            np.array([0]*4 + [1]*4), np.zeros(8, dtype=bool), True)
        cases = {
            "empty blocks": ([], []),
            "overlapping blocks": ([(0, 5), (5, 9)], []),
            "block past the tracking": ([(0, 10)], []),
            "boolean frame": ([(False, 9)], []),
            "float frame": ([(0.0, 9)], []),
            "inverted interval": ([], [{"start_frame": 4, "end_frame": 4}]),
            "interval without frames": ([], [{"code": "GOALS"}]),
            "intervals not a sequence": ([], None),
        }
        for label, module in (("player-attention", player_attention_serving),
                              ("MNIST", shared_temporal_serving)):
            for name, (blocks, intervals) in cases.items():
                with self.subTest(label=label, case=name):
                    with self.assertRaisesRegex(
                            ValueError,
                            f"^{label} requires valid nonoverlapping blocks and tag intervals$"):
                        module.predict(tracking, intervals, blocks)

    def test_single_frame_block_at_frame_zero_is_accepted(self):
        tracking = Tracking(np.zeros((5, 8, 2)), np.zeros((5, 2)),
                            np.array([0]*4 + [1]*4), np.zeros(8, dtype=bool), True)
        seen = []

        def half(received, intervals, artifact, models, *, team_names=None):
            seen.append((len(received.ball), intervals))
            return np.array(["H"])

        labels = player_attention_serving.predict(
            tracking, [{"start_frame": 0, "end_frame": 1}], [(0, 0)],
            half=half, loader=lambda: (object(), []))
        self.assertEqual(labels.tolist(), list("HXXXX"))
        self.assertEqual(seen, [(1, [{"start_frame": 0, "end_frame": 1}])])

    def test_cli_preserves_metadata_order_and_tag_ownership_through_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {name: root / name for name in ("atd_meta", "atd_raw", "smart_tagging")}
            for path in paths.values():
                path.touch()
            paths["atd_meta"].write_text(
                '<Root><Team><Name>Zebra Home</Name></Team>'
                '<Team><Name>Alpha Away</Name></Team></Root>')
            tracks = object()
            intervals = [{"code": "SET PIECES", "start_frame": 0, "end_frame": 3,
                          "tags": {"Team": "alpha-away", "Type": "Corner Kick"}}]

            def infer(received_tracks, received_intervals, blocks, *, team_names):
                self.assertIs(received_tracks, tracks)
                self.assertIs(received_intervals, intervals)
                self.assertEqual(blocks, [(0, 3)])
                self.assertEqual(team_names, ("Zebra Home", "Alpha Away"))
                tags = canonicalize_intervals(received_intervals, team_names)[0]["tags"]
                self.assertEqual(tags, {"Team": "A", "Type": "corner"})
                return np.array(list("HADX"))

            output = root / "synthetic.csv"
            stdout = io.StringIO()
            with patch("possession.run.asset_paths", return_value=paths), \
                    patch("possession.run.load_tracking", return_value=tracks), \
                    patch("possession.run.load_tagging", return_value=intervals), \
                    patch("possession.run.BLOCKS", [(0, 3)]), \
                    patch("possession.shared_temporal_serving.predict", side_effect=infer), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--candidate", "solution", "--out", str(output)]), 0)
            self.assertEqual(json.loads(stdout.getvalue()),
                             {"label_counts": {"H": 1, "A": 1, "D": 1, "X": 1}})
            self.assertEqual(output.read_text(), "frame,label\n0,H\n1,A\n2,D\n3,X\n")


if __name__ == "__main__":
    unittest.main()
