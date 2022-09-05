import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm


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


def create_mlp_model():
    mnist_train = MNIST('./datasets/mnist', train=True, download=True, transform=T.ToTensor())
    mnist_test = MNIST('.datasets/mnist', train=False, download=True, transform=T.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),

        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    while True:
        train(model, train_loader, loss_fn)
        if evaluate(model, test_loader, loss_fn)[1] >= 0.993:
            break

    torch.save(model.state_dict(), 'weights')

    return model


create_mlp_model()
