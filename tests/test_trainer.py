import torch
from torch.utils.data import DataLoader, TensorDataset

from tinyllm.model.gpt import GPT
from tinyllm.training.loss import LanguageModelLoss
from tinyllm.training.optimizer import create_optimizer
from tinyllm.training.trainer import train_epoch, validate


def test_train_epoch_updates_model_parameters() -> None:
    vocab_size = 20
    context_length = 4
    pad_token_id = 0

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=8,
        context_length=context_length,
        num_heads=2,
        num_layers=1,
    )

    x = torch.randint(
        1,
        vocab_size,
        (8, context_length),
    )

    y = torch.randint(
        1,
        vocab_size,
        (8, context_length),
    )

    dataloader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    loss_fn = LanguageModelLoss(
        pad_token_id=pad_token_id,
    )

    optimizer = create_optimizer(
        model,
        learning_rate=1e-2,
    )

    initial_parameter = next(model.parameters()).detach().clone()

    train_loss = train_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    updated_parameter = next(model.parameters()).detach()

    assert train_loss > 0
    assert not torch.equal(
        initial_parameter,
        updated_parameter,
    )


def test_train_epoch_returns_finite_loss() -> None:
    vocab_size = 20
    context_length = 4
    pad_token_id = 0

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=8,
        context_length=context_length,
        num_heads=2,
        num_layers=1,
    )

    x = torch.randint(
        1,
        vocab_size,
        (8, context_length),
    )

    y = torch.randint(
        1,
        vocab_size,
        (8, context_length),
    )

    dataloader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    loss_fn = LanguageModelLoss(
        pad_token_id=pad_token_id,
    )

    optimizer = create_optimizer(
        model,
        learning_rate=1e-2,
    )

    train_loss = train_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert isinstance(train_loss, float)
    assert torch.isfinite(torch.tensor(train_loss))


def test_validate_does_not_update_model_parameters() -> None:
    vocab_size = 20
    context_length = 4
    pad_token_id = 0

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=8,
        context_length=context_length,
        num_heads=2,
        num_layers=1,
    )

    x = torch.randint(
        1,
        vocab_size,
        (8, context_length),
    )

    y = torch.randint(
        1,
        vocab_size,
        (8, context_length),
    )

    dataloader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    loss_fn = LanguageModelLoss(
        pad_token_id=pad_token_id,
    )

    initial_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    validation_loss = validate(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
    )

    final_parameters = list(model.parameters())

    assert validation_loss > 0

    for initial, final in zip(
        initial_parameters,
        final_parameters,
        strict=True,
    ):
        assert torch.equal(initial, final)
