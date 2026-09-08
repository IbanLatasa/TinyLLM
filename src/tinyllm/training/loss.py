from torch import Tensor, nn


class LanguageModelLoss(nn.Module):
    def __init__(self, pad_token_id: int) -> None:
        super().__init__()

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        vocab_size = logits.size(-1)

        return self.loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
