import pytest
import torch

from tinyllm.model.gpt import GPT


def test_gpt_output_shape() -> None:
    batch_size = 2
    sequence_length = 4
    vocab_size = 100

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=16,
        context_length=8,
        num_heads=4,
        num_layers=2,
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

    logits = model(token_ids)

    assert logits.shape == (batch_size, sequence_length, vocab_size)


def test_gpt_rejects_sequence_longer_than_context() -> None:
    model = GPT(
        vocab_size=100, embedding_dim=16, context_length=4, num_heads=4, num_layers=2
    )

    token_ids = torch.randint(0, 100, (2, 5))

    with pytest.raises(ValueError):
        model(token_ids)


def test_gpt_has_expected_number_of_layers() -> None:
    model = GPT(
        vocab_size=100,
        embedding_dim=16,
        context_length=8,
        num_heads=4,
        num_layers=3,
    )

    assert len(model.blocks) == 3
