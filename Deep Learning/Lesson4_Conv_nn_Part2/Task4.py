import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def train(model: nn.Module, train_loader: DataLoader, loss_fn) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = 0

    for x, y in tqdm(train_loader, desc='Train'):
        optimizer.zero_grad()
        output = model(x)

        loss = loss_fn(output, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)

    return train_loss


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn):
    model.eval()

    total_loss = 0
    total = 0
    correct = 0

    for x, y in tqdm(loader, desc='Evaluation'):
        output = model(x)

        loss = loss_fn(output, y)
        total_loss += loss.item()

        _, y_pred = torch.max(output, 1)
        total += y.size(0)
        correct += (y_pred == y).sum().item()

    total_loss /= len(loader)
    accuracy = correct / total

    return total_loss, accuracy


def get_normalize(features: torch.Tensor):
    means = features.data.mean(axis=(0, 1, 2))
    stds = features.data.std(axis=(0, 1, 2))
    return means, stds


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    preds = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        preds.append(torch.argmax(output, dim=1))

    return torch.cat(preds)


def create_simple_conv_cifar():
    train_data = CIFAR10('./data', train=True, transform=T.ToTensor())
    test_data = CIFAR10('./data', train=False, transform=T.ToTensor())
    # means, std = get_normalize(train_data)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 1024),
        nn.ReLU(),

        nn.Linear(1024, 10)
    )

    while True:
        train(model, train_loader, loss_fn)
        if evaluate(model, test_loader, loss_fn)[1] >= 0.7:
            break

    model_preds = predict(model, test_loader, 'cpu')
    torch.save(model_preds, 'model_preds')

    return model


create_simple_conv_cifar()
