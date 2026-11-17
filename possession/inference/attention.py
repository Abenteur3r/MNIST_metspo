"""Masked snapshot attention and team-wise mean/max pooling."""

import numpy as np
from possession.features.players import normalize_tokens


def linear(x, layer):
    return x @ np.asarray(layer["weight"], np.float32).T + np.asarray(layer["bias"], np.float32)


def layer_norm(x, layer):
    mean = x.mean(-1, keepdims=True)
    variance = ((x-mean)**2).mean(-1, keepdims=True)
    return (x-mean)/np.sqrt(variance+np.float32(1e-5))*np.asarray(layer["weight"], np.float32) + np.asarray(layer["bias"], np.float32)


def pooled_tokens(tokens, model):
    output = []
    attention = model["attention"]
    for first in range(0, len(tokens), 512):
        x, mask = normalize_tokens(tokens[first:first+512], model["token_mean"], model["token_scale"])
        for layer in model["player_layers"]:
            x = np.maximum(linear(x, layer), 0)
        length, _, slots, width = x.shape
        x = x.reshape(length, 2*slots, width)
        valid = mask.reshape(length, 2*slots)
        qkv = linear(x, attention["qkv"]).reshape(length, 2*slots, 3, 2, 16)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        score = (q @ k.transpose(0, 1, 3, 2)) * np.float32(.25)
        teams = np.repeat(np.arange(2), slots)
        same = teams[:, None] == teams[None, :]
        score += np.asarray(attention["team_bias"], np.float32)[None, :, None, None]*same[None, None]
        score = np.where(valid[:, None, None, :], score, np.float32(-1e9))
        weights = np.exp(score-score.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(length, 2*slots, width)
        x = layer_norm(x+linear(attended, attention["projection"]), attention["norm1"])
        feedforward = linear(np.maximum(linear(x, attention["ff1"]), 0), attention["ff2"])
        x = layer_norm(x+feedforward, attention["norm2"]).reshape(length, 2, slots, width)
        count = mask.sum(-1, keepdims=True)
        mean = np.where(mask[..., None], x, 0).sum(-2)/np.maximum(count, 1)
        maximum = np.where(mask[..., None], x, -np.inf).max(-2)
        maximum = np.where(count > 0, maximum, 0)
        output.append(np.concatenate((mean, maximum), axis=-1).reshape(length, -1).astype(np.float32))
    return np.concatenate(output)
