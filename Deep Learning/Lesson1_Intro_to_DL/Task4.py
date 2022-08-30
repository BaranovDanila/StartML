import torch
from torch import nn


def function04(x: torch.Tensor, y: torch.Tensor, step: float = 1e-2):
    layer = nn.Linear(in_features=x.shape[1], out_features=(1 if y.dim() == 1 else y.shape[1]))

    while True:
        y_predicted = layer(x).ravel()
        mse = torch.mean((y_predicted - y) ** 2)
        print(mse)
        if mse < 0.3:
            return layer

        mse.backward()

        with torch.no_grad():
            layer.weight -= step * layer.weight.grad
            layer.bias -= step * layer.bias.grad

        layer.zero_grad()


n_features = 2
n_objects = 300

w_true = torch.randn(n_features)
X = (torch.rand(n_objects, n_features) - 0.5) * 5
Y = X @ w_true + torch.randn(n_objects) / 2

function04(X, Y)
