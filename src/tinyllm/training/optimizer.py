from torch import nn
from torch.optim import AdamW, Optimizer


def create_optimizer(
    model: nn.Module, learning_rate: float = 3e-4, weight_decay: float = 0.01
) -> Optimizer:
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
