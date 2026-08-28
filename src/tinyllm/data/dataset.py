import torch
from torch import Tensor
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        stories: list[list[int]],
        context_length: int,
        eos_id: int,
        bos_id: int,
        pad_id: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("Context length must be greater that 0")

        self.context_length = context_length
        self.pad_id = pad_id
        self.chunks: list[list[int]] = []

        for story in stories:
            tokens = [bos_id, *story, eos_id]

            for start in range(0, len(tokens) - 1, self.context_length):
                chunk = tokens[start : start + context_length + 1]

                if len(chunk) < context_length + 1:
                    padding_lenght = context_length + 1 - len(chunk)
                    chunk += [pad_id] * padding_lenght

                self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        chunk = self.chunks[index]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y
