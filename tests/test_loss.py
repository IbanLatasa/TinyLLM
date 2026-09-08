import torch

from tinyllm.training.loss import LanguageModelLoss


def test_language_model_loss_returns_scalar() -> None:
    batch_size = 2
    sequence_length = 4
    vocab_size = 10
    pad_token_id = 3

    logits = torch.randn(batch_size, sequence_length, vocab_size)

    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))

    loss_fn = LanguageModelLoss(pad_token_id=pad_token_id)

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0


def test_language_model_loss_is_finite() -> None:
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))

    loss_fn = LanguageModelLoss(pad_token_id=3)

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)


def test_language_model_loss_ignores_padding() -> None:
    pad_token_id = 4

    logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0, 0.0]]])

    targets = torch.tensor([[0, pad_token_id]])

    loss_fn = LanguageModelLoss(pad_token_id=pad_token_id)

    loss = loss_fn(logits, targets)

    assert loss.item() < 0.01
