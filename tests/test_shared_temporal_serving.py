"""The packaged MNIST ensemble contract, pinned against the real artifact."""

import copy
import unittest

from possession import shared_temporal_serving as serving

REJECTION = "^Invalid selected MNIST ensemble, original members or external provenance$"


class MnistEnsembleValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact, cls.models = serving.load_model.__wrapped__()

    def artifact_with(self, **changes):
        artifact = copy.deepcopy(self.artifact)
        artifact.update(changes)
        return artifact

    def assert_rejected(self, artifact, models=None):
        with self.assertRaisesRegex(ValueError, REJECTION):
            serving.validate(artifact, models if models is not None else copy.deepcopy(self.models))

    def test_accepts_the_packaged_ensembles(self):
        serving.validate(self.artifact, self.models)

    def test_rejects_identity_fields(self):
        cases = {
            "schema": "wrong",
            "candidate": "wrong",
            "feature_schema": "wrong",
            "token_names": [],
            "selection_protocol": "wrong",
            "campaign_selection_protocol": "wrong",
            "training_freezes": {},
            "task_isolation": "wrong",
            "arm": "wrong",
            "decoder": {},
            "training_matches": [],
            "seeds": [],
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                self.assert_rejected(self.artifact_with(**{key: value}))

    def test_rejects_flag_fields(self):
        for key in ("target_data_used", "public_gt_used"):
            with self.subTest(key=key):
                self.assert_rejected(self.artifact_with(**{key: True}))
        for key in serving_selection_flags(self.artifact):
            with self.subTest(key=key):
                self.assert_rejected(self.artifact_with(**{key: False}))

    def test_rejects_freeze_commits_that_are_not_hex(self):
        for key in ("evaluation_freeze_commit", "transfer_freeze_commit", "campaign_selection_freeze_commit"):
            with self.subTest(key=key):
                self.assert_rejected(self.artifact_with(**{key: "not-a-commit"}))

    def test_rejects_feature_names(self):
        names = list(self.artifact["feature_names"])
        cases = {
            "length": names[:-1],
            "duplicate": names[:-1] + [names[0]],
            "empty": names[:-1] + [""],
            "tag suffix": names[:-8] + list(reversed(names[-8:])),
        }
        for label, value in cases.items():
            with self.subTest(label=label):
                self.assert_rejected(self.artifact_with(feature_names=value))

    def test_rejects_member_manifests(self):
        artifact = copy.deepcopy(self.artifact)
        models = copy.deepcopy(self.models)
        with self.subTest("member count"):
            self.assert_rejected(self.artifact_with(members=artifact["members"][:-1]))
        with self.subTest("model count"):
            self.assert_rejected(copy.deepcopy(self.artifact), models[:-1])
        with self.subTest("member order"):
            members = copy.deepcopy(artifact["members"])
            members[0], members[1] = members[1], members[0]
            self.assert_rejected(self.artifact_with(members=members))

    def test_rejects_member_fields(self):
        cases = {
            "file": "wrong.json",
            "training_matches": [],
            "normalization_matches": [],
            "training_freeze_commit": "wrong",
            "source_experiment": "wrong",
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                artifact = copy.deepcopy(self.artifact)
                artifact["members"][0][key] = value
                self.assert_rejected(artifact)

    def test_rejects_traversal_member_filenames(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["members"][0]["file"] = "../" + artifact["members"][0]["file"]
        self.assert_rejected(artifact)

    def test_rejects_model_fields(self):
        cases = {
            "source_experiment": "wrong",
            "training_matches": [],
            "seed": 0,
            "schema": "wrong",
            "kind": "wrong",
            "arm": "wrong",
            "normalization_matches": [],
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                models = copy.deepcopy(self.models)
                models[0][key] = value
                self.assert_rejected(copy.deepcopy(self.artifact), models)

    def test_requires_the_freeze_commit_outside_the_pilot_matches(self):
        models = copy.deepcopy(self.models)
        models[3].pop("training_freeze_commit")
        self.assert_rejected(copy.deepcopy(self.artifact), models)

    def test_tolerates_the_freeze_commit_of_a_pilot_match(self):
        models = copy.deepcopy(self.models)
        models[0]["training_freeze_commit"] = "wrong"
        serving.validate(copy.deepcopy(self.artifact), models)

    def test_rejects_a_member_whose_parameters_fail(self):
        models = copy.deepcopy(self.models)
        models[0]["player_layers"] = []
        self.assert_rejected(copy.deepcopy(self.artifact), models)


def serving_selection_flags(artifact):
    return [key for key in artifact if key.startswith("accepted_") or key == "selected_by_campaign"]


if __name__ == "__main__":
    unittest.main()
