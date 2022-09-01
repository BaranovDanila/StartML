import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x)

        loss = loss_fn(output, x)
        loss.backward()
        print(round(loss, 5))

        optimizer.step()

        total_loss += loss

    total_loss /= len(data_loader)

    return total_loss
