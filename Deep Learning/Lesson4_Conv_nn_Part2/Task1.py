import torch


def get_normalize(features: torch.Tensor):
    means = features.mean(axis=(0, 2, 3))
    stds = features.std(axis=(0, 2, 3))
    return means, stds
