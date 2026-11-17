"""NumPy-only inference for original shared-readout discovery/continuation members."""

import numpy as np
from possession.features.ownership import normalize
from possession.inference.attention import pooled_tokens

SCHEMA = "r37-task-isolated-v1"
KIND = "r37_task_isolation"
ARMS = ("shared_readout",)
TASK_ISOLATION = {
    "paths": 2, "path_width": 16, "dilations": [1, 2, 4, 8, 16],
    "kernel_size": 3, "activation": "relu", "anchor_hz": 5,
    "receptive_field_anchors": 63, "halo_anchors_each_side": 31,
    "binary_order": ["dead_logit", "home_given_live_logit"],
    "initialization": "construct_original_r7_then_two_new_paths_and_readout",
    "snapshot": "unchanged_r7_attention_mean_max_shared_between_tasks",
    "cross_readout_initialization": "zero_in_both_arms",
    "isolated": "dead_reads_first16_owner_reads_last16",
    "shared_readout": "both_heads_can_learn_all32_connections",
    "masked_readout_parameters": 32,
    "loss": "dead_BCE_plus_live_masked_owner_BCE_mean_over_all_anchors",
    "class_order": ["H", "A", "D"], "dead_owner_supervision": False,
    "normalization": "original_R7_excluding_fold_seed_unchanged",
}


def _array(value, shape, *, positive=False):
    array = np.asarray(value, np.float32)
    if array.shape != shape or not np.isfinite(array).all() or (positive and not (array > 0).all()):
        raise ValueError("Invalid shared-temporal parameter shape or finite-value contract")
    return array


_ATTENTION_SHAPES = {"qkv": ((96, 32), (96,)), "projection": ((32, 32), (32,)),
                     "norm1": ((32,), (32,)), "ff1": ((64, 32), (64,)),
                     "ff2": ((32, 64), (32,)), "norm2": ((32,), (32,))}


def _recipe_contract(model):
    return (model.get("schema") == SCHEMA and model.get("kind") == KIND
            and model.get("arm") in ARMS and model.get("task_isolation") == TASK_ISOLATION
            and model.get("class_order") == ["H", "A", "D"]
            and model.get("target_data_used") is False and model.get("public_gt_used", False) is False)


def _statistics_contract(model):
    for prefix, size in (("", 119), ("token_", 10)):
        _array(model[prefix+"mean"], (size,))
        _array(model[prefix+"scale"], (size,), positive=True)
    return True


def _player_contract(model):
    if len(model["player_layers"]) != 2:
        return False
    for index, layer in enumerate(model["player_layers"]):
        _array(layer["weight"], (32, 20 if index == 0 else 32))
        _array(layer["bias"], (32,))
    return True


def _attention_contract(model):
    attention = model["attention"]
    if set(attention) != set(_ATTENTION_SHAPES) | {"team_bias"}:
        return False
    for name, (weight, bias) in _ATTENTION_SHAPES.items():
        _array(attention[name]["weight"], weight)
        _array(attention[name]["bias"], bias)
    _array(attention["team_bias"], (2,))
    return True


def _paths_contract(model):
    if len(model["paths"]) != 2:
        return False
    for path in model["paths"]:
        if len(path) != 5:
            return False
        for index, layer in enumerate(path):
            _array(layer["weight"], (16, 366 if index == 0 else 16, 3))
            _array(layer["bias"], (16,))
            if layer["dilation"] != 2**index or layer["relu"] is not True:
                return False
    return True


def _readout_contract(model):
    weights = _array(model["readout"]["weight"], (2, 32))
    _array(model["readout"]["bias"], (2,))
    return (model["arm"] != "isolated"
            or not (np.any(weights[0, 16:] != 0) or np.any(weights[1, :16] != 0)))


def _parameters(model):
    try:
        valid = (_recipe_contract(model) and _statistics_contract(model)
                 and _player_contract(model) and _attention_contract(model)
                 and _paths_contract(model) and _readout_contract(model))
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError("shared-temporal artifact violates its fixed temporal-task recipe") from None
    if not valid:
        raise ValueError("shared-temporal artifact violates its fixed temporal-task recipe")


def _clock_contract(values, tokens):
    return (values.ndim == 2 and values.shape[1] == 119
            and tokens.ndim == 4 and tokens.shape[1] == 2
            and tokens.shape[2] >= 1 and tokens.shape[3] == 10
            and len(values) == len(tokens))


def _convolved(x, layer):
    weight, bias = np.asarray(layer["weight"], np.float32), np.asarray(layer["bias"], np.float32)
    dilation = layer["dilation"]
    padded = np.pad(x, ((dilation, dilation), (0, 0)))
    result = np.broadcast_to(bias, (len(x), 16)).copy()
    for kernel in range(3):
        result += padded[kernel*dilation:kernel*dilation+len(x)] @ weight[:, :, kernel].T
    return np.maximum(result, 0)


def _path(inputs, layers):
    x = inputs
    for layer in layers:
        x = _convolved(x, layer)
    return x


def logits(values, tokens, model):
    """Return the two raw binary logits; no truth, identity or fit operations."""
    _parameters(model)
    if not _clock_contract(values, tokens):
        raise ValueError("shared-temporal feature schema or clock mismatch")
    if not len(values):
        return np.empty((0, 2), np.float32)
    inputs = np.concatenate((normalize(values, model["mean"], model["scale"]), pooled_tokens(tokens, model)), axis=-1)
    hidden = np.concatenate([_path(inputs, layers) for layers in model["paths"]], axis=1)
    return hidden @ np.asarray(model["readout"]["weight"], np.float32).T + np.asarray(model["readout"]["bias"], np.float32)


def binary_probability(binary_logits):
    z = np.asarray(binary_logits, np.float64)
    if z.ndim != 2 or z.shape[1] != 2 or not np.isfinite(z).all():
        raise ValueError("shared-temporal requires finite two-column binary logits")
    live = np.exp(-np.logaddexp(0., z[:, 0]))
    dead = np.exp(-np.logaddexp(0., -z[:, 0]))
    home = np.exp(-np.logaddexp(0., -z[:, 1]))
    away = np.exp(-np.logaddexp(0., z[:, 1]))
    return np.stack((live*home, live*away, dead), axis=1)


def probability(values, tokens, models):
    if not models or len({model.get("arm") for model in models}) != 1:
        raise ValueError("shared-temporal requires a nonempty bag from one registered arm")
    total = np.zeros((len(values), 3), np.float64)
    for model in models:
        total += binary_probability(logits(values, tokens, model))
    return total / len(models)
