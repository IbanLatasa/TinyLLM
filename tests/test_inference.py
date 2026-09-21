from pathlib import Path

import torch

from tinyllm.inference.generate import generate_greedy
from tinyllm.training.checkpoint import load_model_checkpoint


def test_greedy_generation_adds_tokens() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_checkpoint(
        Path("artifacts/checkpoints/latest.pt"), device
    )

    input_ids = torch.tensor(
        [[1, 2]],
        dtype=torch.long,
        device=device,
    )

    output = generate_greedy(model, input_ids, 3, 8)

    assert output.shape == (1, 5)
