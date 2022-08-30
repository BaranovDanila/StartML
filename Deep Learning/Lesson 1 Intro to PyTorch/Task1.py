import torch


def function01(tensor: torch.tensor, count_over: str):
    if count_over == 'columns':
        return torch.mean(tensor, dim=1)

    if count_over == 'rows':
        return torch.mean(tensor, dim=0)

    return None
