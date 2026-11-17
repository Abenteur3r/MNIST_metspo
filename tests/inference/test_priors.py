import copy
import unittest
import numpy as np
from possession.inference.pipeline import load_model, validate
from possession.inference.contracts import MATCHES as HELD


class DecoderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact, cls.models = load_model()

    def assertRejected(self, artifact):
        with self.assertRaises(ValueError):
            validate(artifact, self.models)

    def test_decoder_contract_rejections(self):
        cases = [("stride", ("control", "stride", 1)),
                 ("window", ("control", "window", 15)),
                 ("negative control distance", ("control", "control_distance", -1.0)),
                 ("non integral confirmation", ("control", "confirmation", 3.0)),
                 ("zero restart window", ("control", "restart_window", 0)),
                 ("dead mix", ("decoder", "dead_mix", 0.5)),
                 ("confidence range", ("decoder", "confidence", 1.5)),
                 ("quantile range", ("decoder", "min_quantile", -0.1)),
                 ("negative penalty", ("decoder", "switch_penalty", -1.0)),
                 ("negative duration weight", ("decoder", "duration_weight", -0.1))]
        for label, (section, key, value) in cases:
            with self.subTest(label):
                artifact = copy.deepcopy(self.artifact)
                artifact[section][key] = value
                self.assertRejected(artifact=artifact)
        priors = [("training matches", "training_matches", list(HELD[:-1])),
                  ("transition shape", "transition", np.ones((2, 2))),
                  ("initial shape", "initial", np.ones(2)),
                  ("transition sum", "transition", np.full((3, 3), 0.5)),
                  ("initial sum", "initial", np.full(3, 0.33)),
                  ("negative prior", "initial", np.array([-1.0, 1.0, 1.0])),
                  ("duration count", "durations", [np.array([1.0]), np.array([1.0])]),
                  ("empty duration", "durations", [np.array([1.0]), np.array([]), np.array([1.0])]),
                  ("infinite duration", "durations", [np.array([1.0]), np.array([np.inf]), np.array([1.0])]),
                  ("zero duration", "durations", [np.array([1.0]), np.array([0.0]), np.array([1.0])])]
        for label, key, value in priors:
            with self.subTest("priors " + label):
                artifact = copy.deepcopy(self.artifact)
                artifact["priors"][key] = value
                self.assertRejected(artifact=artifact)
