import torch

from tinyllm.model.block import TransformerBlock


def test_transformer_block_preserves_shape() -> None:
    batch_size = 2
    sequence_length = 4
    embedding_dim = 8

    block = TransformerBlock(embedding_dim, num_heads=2, context_length=sequence_length)

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    output = block(x)

    assert output.shape == x.shape
