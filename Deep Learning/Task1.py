import torch


def function01(tensor: torch.tensor, count_over: str):
    if count_over == 'columns':
        return tensor.mean(dim=0)

    if count_over == 'rows':
        return tensor.mean(dim=1)

    return None
