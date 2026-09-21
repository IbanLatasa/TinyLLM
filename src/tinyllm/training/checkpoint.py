from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from tinyllm.model.gpt import GPT


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    train_loss: float,
    validation_loss: float,
    model_config: dict[str, int],
    scaler_state_dict: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler_state_dict,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "model_config": model_config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: Optimizer, device: torch.device
) -> dict:
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[GPT, dict]:
    checkpoint = torch.load(path, map_location=device)

    model_config = checkpoint["model_config"]

    model = GPT(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        context_length=model_config["context_length"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    return model, checkpoint
