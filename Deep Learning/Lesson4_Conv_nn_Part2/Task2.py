import torch
import torchvision.transforms as T


def get_normalize(features: torch.Tensor):
    means = features.mean(axis=(0, 1, 2))
    stds = features.std(axis=(0, 1, 2))
    return means, stds


def get_augmentations(train: bool = True) -> T.Compose:
    means = [125.30691805, 122.95039414, 113.86538318]
    stds = [62.99321928, 62.08870764, 66.70489964]

    train_transform = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
    )
    test_transform = T.Compose(
        [
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=means, std=stds)
        ]
    )
    return train_transform if train else test_transform
