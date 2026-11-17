"""All XML and ATD below are tiny, invented fixtures, never match extracts."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession.data.match import asset_paths
from possession.data.tracking import load_tracking
from possession.data.tagging import load_tagging
from possession.run import main, write_submission


META = """<Root><Team id="home"><ProviderParameter>
<Name>attack_direction_first_half</Name><Value>left_to_right</Value>
</ProviderParameter></Team><Team id="away"/>
<Player teamId="home"><ProviderParameter><Name>position_type</Name>
<Value>Goalkeeper</Value></ProviderParameter></Player><Player teamId="away"/>
</Root>"""
RAW = "1:0.1,0.2,0;0.9,0.8,0:0.4,0.5,0\n3:NaN,NaN,NaN;0.8,0.7,0:NaN,NaN,NaN\n"


class InputOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.meta = self.root / "meta.xml"
        self.raw = self.root / "raw.txt"
        self.meta.write_text(META)
        self.raw.write_text(RAW)

    def test_atd_counter_precision_scaling_and_metadata(self):
        tracking = load_tracking(self.raw, self.meta, n_frames=4, pitch=(100., 50.))
        self.assertEqual(tracking.players.dtype, np.float32)
        self.assertEqual(tracking.ball.dtype, np.float32)
        np.testing.assert_allclose(tracking.players[0], [[10., 10.], [90., 40.]])
        np.testing.assert_allclose(tracking.ball[0], [40., 25.])
        self.assertTrue(np.isnan(tracking.players[1]).all())
        self.assertTrue(np.isnan(tracking.ball[2:]).all())
        np.testing.assert_array_equal(tracking.teams, [0, 1])
        np.testing.assert_array_equal(tracking.keepers, [True, False])
        self.assertTrue(tracking.home_defends_left)

    def test_reverse_attack_direction(self):
        self.meta.write_text(META.replace("left_to_right", "right_to_left"))
        self.assertFalse(load_tracking(self.raw, self.meta, n_frames=4).home_defends_left)

    def test_malformed_tracking_is_rejected_without_echoing_input(self):
        sentinel = "PRIVATE_INPUT_SENTINEL"
        first = RAW.splitlines()[0] + "\n"
        cases = [first + first, first.replace("1:", "0:"), first.replace("1:", "5:"),
                 first.replace("0.1", sentinel), "1:0,0,0:0,0,0\n"]
        for content in cases:
            with self.subTest(case=cases.index(content)):
                self.raw.write_text(content)
                with self.assertRaises(ValueError) as caught:
                    load_tracking(self.raw, self.meta, n_frames=4)
                self.assertNotIn(sentinel, str(caught.exception))

    def test_invalid_metadata_is_rejected_without_echoing_input(self):
        self.meta.write_text(META.replace("left_to_right", "PRIVATE_INPUT_SENTINEL"))
        with self.assertRaisesRegex(ValueError, "Invalid tactical metadata") as caught:
            load_tracking(self.raw, self.meta, n_frames=4)
        self.assertNotIn("PRIVATE_INPUT_SENTINEL", str(caught.exception))

    def test_tag_rounding_and_repeated_team_labels(self):
        path = self.root / "tags.xml"
        path.write_text('''<root>
<instance><start>0.06</start><end>0.18</end><code>SET PIECES</code>
<label><group>Team</group><text>first</text></label>
<label><group>Team</group><text>N/A</text></label></instance>
<instance><start>0.16</start><end>0.28</end><code>GOALS</code></instance>
<instance><start>0.32</start><end>0.4</end><code>BUILD UP</code></instance>
<instance><start>0.32</start><end>0.4</end><code>set pieces</code></instance>
</root>''')
        rows = load_tagging(path)
        self.assertEqual((rows[0]["start_frame"], rows[0]["end_frame"]), (2, 4))
        self.assertEqual(rows[0]["tags"]["Team"], "N/A")
        self.assertEqual((rows[1]["start_frame"], rows[1]["end_frame"]), (4, 7))
        self.assertEqual([row["code"] for row in rows],
                         ["SET PIECES", "GOALS", "BUILD UP", "set pieces"])

    def test_bad_tag_time_is_rejected_without_echoing_input(self):
        path = self.root / "bad.xml"
        path.write_text("<root><instance><start>PRIVATE_INPUT_SENTINEL</start></instance></root>")
        with self.assertRaisesRegex(ValueError, "Invalid Smart Tagging") as caught:
            load_tagging(path)
        self.assertNotIn("PRIVATE_INPUT_SENTINEL", str(caught.exception))

    def test_csv_exact_bytes_and_no_overwrite(self):
        path = self.root / "nested" / "output.csv"
        write_submission(path, np.array(list("HADX")))
        expected = b"frame,label\n0,H\n1,A\n2,D\n3,X\n"
        self.assertEqual(path.read_bytes(), expected)
        with self.assertRaises(FileExistsError):
            write_submission(path, ["D"])
        self.assertEqual(path.read_bytes(), expected)

    def test_dataset_location_is_explicit_and_resolved_at_call_time(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MTAH_DATA_DIR"):
                asset_paths()
        with patch.dict(os.environ, {"MTAH_DATA_DIR": str(self.root)}):
            self.assertEqual(asset_paths()["atd_raw"].parent, self.root / "tactical")
            self.assertEqual(asset_paths(self.root / "other")["atd_raw"].parent,
                             self.root / "other" / "tactical")

    def test_cli_does_not_echo_exception_payload(self):
        with patch("possession.run.regenerate", side_effect=ValueError("PRIVATE_INPUT_SENTINEL")):
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                main(["--out", str(self.root / "out.csv")])
            self.assertEqual(caught.exception.code, 1)
            self.assertNotIn("PRIVATE_INPUT_SENTINEL", error.getvalue())

    def test_installed_cli_from_unrelated_directory_on_synthetic_inputs(self):
        folder = self.root / "tactical"
        folder.mkdir()
        meta = '<Root><Team id="home"><Name>Home Example</Name><ProviderParameter>' \
               '<Name>attack_direction_first_half</Name><Value>left_to_right</Value>' \
               '</ProviderParameter></Team><Team id="away"><Name>Away Example</Name></Team>'
        meta += ''.join(f'<Player teamId="{team}"/>' for team in ["home"] * 4 + ["away"] * 4)
        (folder / "BAR-ATH_tactical_atd.xml").write_text(meta + '</Root>')
        raw = ''.join(f'{frame}:' + ';'.join(['0.3,0.4,0'] * 4 + ['0.7,0.6,0'] * 4)
                      + ':0.5,0.5,0\n' for frame in range(1, 21))
        (folder / "BAR-ATH_tactical_atd.txt").write_text(raw)
        (folder / "BAR-ATH_tactical_smart_tagging.xml").write_text("<root/>")
        output = self.root / "output.csv"
        env = dict(os.environ, MTAH_DATA_DIR=str(self.root))
        # Shrink only the match dimensions; use the installed real ensemble and CLI.
        script = ('from functools import partial; import possession.run as run; '
                  'run.BLOCKS = ((1, 8), (11, 18)); '
                  'run.load_tracking = partial(run.load_tracking, n_frames=20); '
                  'run.main()')
        command = [sys.executable, "-I", "-c", script, "--out", str(output)]
        result = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        counts = json.loads(result.stdout)["label_counts"]
        self.assertEqual(counts["X"], 4)
        self.assertEqual(sum(counts[label] for label in "HAD"), 16)
        # All data here is synthetic; checking exact frame indexing is unrestricted.
        rows = output.read_text().splitlines()
        self.assertEqual(rows[0], "frame,label")
        self.assertEqual(len(rows), 21)
        for first, last in ((1, 8), (11, 18)):
            self.assertEqual(rows[first], f"{first - 1},X")
            self.assertIn(rows[first + 1], [f"{first},{label}" for label in "HAD"])
            self.assertIn(rows[last + 1], [f"{last},{label}" for label in "HAD"])
            self.assertEqual(rows[last + 2], f"{last + 1},X")
        # An existing result cannot be overwritten even on a second CLI invocation.
        again = subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(again.returncode, 1)
        self.assertIn("already exist", again.stderr)
