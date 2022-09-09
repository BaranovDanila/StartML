import torch
# import numpy as np
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    preds = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        preds.append(torch.argmax(output, dim=1))

    return torch.cat(preds)
