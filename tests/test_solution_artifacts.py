"""Validate frozen serving assets without reading any competition data."""

import copy
import unittest

from possession.shared_temporal_serving import load_model, validate


class SolutionArtifactTests(unittest.TestCase):
    def test_complete_ensemble_loads_with_external_provenance(self):
        artifact, models = load_model()
        self.assertEqual(artifact["candidate"], "solution")
        self.assertEqual(len(models), 21)
        self.assertEqual(len({member["file"] for member in artifact["members"]}), 21)
        self.assertFalse(artifact["target_data_used"])
        self.assertFalse(artifact["public_gt_used"])

    def test_incomplete_or_unsafe_member_manifest_is_rejected(self):
        artifact, models = load_model()
        for filename in ("../outside.json", "missing.json"):
            changed = copy.deepcopy(artifact)
            changed["members"][0]["file"] = filename
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                validate(changed, models)
        with self.assertRaises(ValueError):
            validate(artifact, models[:-1])

    def test_target_label_provenance_is_rejected(self):
        artifact, models = load_model()
        for field in ("target_data_used", "public_gt_used"):
            changed = dict(artifact, **{field: True})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(changed, models)
