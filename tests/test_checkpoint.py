from pathlib import Path

import torch
from torch import nn

from tinyllm.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from tinyllm.training.optimizer import create_optimizer


def test_checkpoint_restores_model(
    tmp_path: Path,
) -> None:
    model = nn.Linear(4, 2)
    optimizer = create_optimizer(model)

    original_weight = model.weight.detach().clone()

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        train_loss=1.5,
        validation_loss=1.7,
    )

    with torch.no_grad():
        model.weight.add_(10)

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        model.weight,
        original_weight,
    )


def test_checkpoint_restores_metadata(
    tmp_path: Path,
) -> None:
    model = nn.Linear(4, 2)
    optimizer = create_optimizer(model)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        train_loss=2.3,
        validation_loss=2.5,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert checkpoint["epoch"] == 5
    assert checkpoint["train_loss"] == 2.3
    assert checkpoint["validation_loss"] == 2.5
