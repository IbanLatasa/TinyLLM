from torch import Tensor, nn

from tinyllm.model.block import TransformerBlock
from tinyllm.model.embeddings import GPTEmbeddings


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.context_length = context_length

        self.embeddings = GPTEmbeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_form = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        _, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("sequence_length cannot exceed context_length")

        x = self.embeddings(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_form(x)

        logits = self.lm_head(x)

        return logits
