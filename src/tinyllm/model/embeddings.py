import torch
from torch import Tensor, nn


class GPTEmbeddings(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, context_length: int
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        _, sequence_length = token_ids.shape

        positions = torch.arange(sequence_length, device=token_ids.device)

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings
