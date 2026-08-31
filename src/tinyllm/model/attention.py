import math

import torch
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    def __init__(self, head_dim: int, context_length: int, embedding_dim: int) -> None:
        super().__init__()

        self.head_dim = head_dim

        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        mask = torch.tril(torch.ones(context_length, context_length))

        self.register_buffer("causal_mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        _, sequence_length, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = self.causal_mask[:sequence_length, :sequence_length]

        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)

        return attention_weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, context_length: int) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(
                    head_dim=head_dim,
                    embedding_dim=embedding_dim,
                    context_length=context_length,
                )
                for _ in range(num_heads)
            ]
        )

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        head_outputs = [head(x) for head in self.heads]

        output = torch.cat(head_outputs, dim=-1)

        return self.output_projection(output)
