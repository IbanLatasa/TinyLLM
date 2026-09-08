import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = loss_fn(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_fn(logits, y)

            total_loss += loss.item()

    return total_loss / len(dataloader)
