from torch import Tensor, nn

from tinyllm.model.attention import MultiHeadAttention
from tinyllm.model.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            context_length=context_length,
        )

        self.ln2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim=embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
