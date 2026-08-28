import torch

from tinyllm.data.dataset import LanguageModelDataset


def test_dataset_returns_correct_shapes() -> None:
    dataset = LanguageModelDataset(
        stories=[[10, 20, 30]], context_length=4, eos_id=1, bos_id=2, pad_id=3
    )

    x, y = dataset[0]

    assert x.shape == torch.Size([4])
    assert y.shape == torch.Size([4])


def test_dataset_targets_are_shifted() -> None:
    dataset = LanguageModelDataset(
        stories=[[10, 20, 30]], context_length=4, eos_id=1, bos_id=2, pad_id=3
    )

    x, y = dataset[0]

    assert torch.equal(x[1:], y[:-1])


def test_dataset_returns_long_tensors() -> None:
    dataset = LanguageModelDataset(
        stories=[[10, 20, 30]], context_length=4, eos_id=1, bos_id=2, pad_id=3
    )

    x, y = dataset[0]

    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_adds_padding() -> None:
    dataset = LanguageModelDataset(
        stories=[[10, 20]], context_length=4, eos_id=1, bos_id=2, pad_id=3
    )

    x, y = dataset[0]

    assert torch.equal(x, torch.tensor([2, 10, 20, 1]))
    assert torch.equal(y, torch.tensor([10, 20, 1, 3]))
