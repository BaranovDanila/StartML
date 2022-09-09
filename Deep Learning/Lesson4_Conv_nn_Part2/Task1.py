import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms as T


def get_normalize(features: torch.Tensor):
    means = features.data.mean(axis=(0, 1, 2))
    stds = features.data.std(axis=(0, 1, 2))
    return means, stds


train_data = CIFAR10('./data', train=True, transform=T.ToTensor(), download=True)
test_data = CIFAR10('./data', train=False, transform=T.ToTensor(), download=True)

means, stds = get_normalize(train_data)
print(f'Means: {means}')
print(f'Stds: {stds}')
