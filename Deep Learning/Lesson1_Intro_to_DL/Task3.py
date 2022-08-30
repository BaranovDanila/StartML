import torch


def function02(tensor: torch.tensor):
    weights = torch.rand(tensor.shape[-1], dtype=torch.float32, requires_grad=True)
    return weights


def function03(x: torch.tensor, y: torch.tensor):
    step = 1e-2
    weights = function02(x)

    while True:
        y_predicted = torch.matmul(x, weights)
        mse = torch.mean((y - y_predicted) ** 2)

        if mse < 1:
            break

        mse.backward()

        with torch.no_grad():
            weights -= step * weights.grad

        weights.grad.zero_()

    return weights


