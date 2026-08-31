import pytest
import torch

from tinyllm.model.attention import CausalSelfAttention, MultiHeadAttention


def test_attention_output_shape() -> None:
    batch_size = 2
    sequence_length = 4
    embedding_dim = 8
    head_dim = 4

    attention = CausalSelfAttention(
        head_dim=head_dim, context_length=sequence_length, embedding_dim=embedding_dim
    )

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    output = attention(x)

    assert output.shape == (batch_size, sequence_length, head_dim)


def test_attention_cant_see_future_tokens() -> None:
    torch.manual_seed(42)

    attention = CausalSelfAttention(head_dim=4, context_length=4, embedding_dim=8)

    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()

    x2[:, -1, :] = torch.randn(8)

    output1 = attention(x1)
    output2 = attention(x2)

    assert torch.allclose(output1[:, :-1, :], output2[:, :-1, :], atol=1e-6)


def test_multi_head_attention_preserves_shape() -> None:
    batch_size = 2
    sequence_length = 4
    embedding_dim = 8

    attention = MultiHeadAttention(
        num_heads=2, embedding_dim=embedding_dim, context_length=sequence_length
    )

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    output = attention(x)

    assert x.shape == output.shape


def test_multi_head_attention_requires_divisible_dimensions() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(
            embedding_dim=10,
            num_heads=3,
            context_length=4,
        )
