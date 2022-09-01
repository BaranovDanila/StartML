import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()
    total_loss = 0

    for x, y in data_loader:
        y_predicted = model(x)
        total_loss += loss_fn(y_predicted, y)

    return total_loss / len(data_loader)
