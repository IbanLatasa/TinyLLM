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

    model_config = {
        "in_features": 4,
        "out_features": 2,
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=100,
        train_loss=1.5,
        validation_loss=1.7,
        model_config=model_config,
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

    model_config = {
        "in_features": 4,
        "out_features": 2,
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=100,
        train_loss=2.3,
        validation_loss=2.5,
        model_config=model_config,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert checkpoint["step"] == 100
    assert checkpoint["train_loss"] == 2.3
    assert checkpoint["validation_loss"] == 2.5
    assert checkpoint["model_config"] == model_config