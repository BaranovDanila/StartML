import torch
from torch import nn


def function04(x: torch.Tensor, y: torch.Tensor, step: float = 1e-2):
    layer = nn.Linear(in_features=x.shape[1], out_features=y.shape[0])

    while True:
        y_predicted = layer(x)
        mse = torch.mean((y_predicted - y) ** 2)

        if mse < 0.3:
            return layer

        mse.backward()

        with torch.no_grad():
            layer.weight -= step * layer.weight.grad

        layer.zero_grad()
