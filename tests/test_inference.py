import torch

from tinyllm.inference.generate import generate_greedy
from tinyllm.model.gpt import GPT


def test_greedy_generation_adds_tokens() -> None:
    device = torch.device("cpu")

    vocab_size = 20
    context_length = 8

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=8,
        context_length=context_length,
        num_heads=2,
        num_layers=1,
    ).to(device)

    input_ids = torch.tensor(
        [[1, 2]],
        dtype=torch.long,
        device=device,
    )

    output = generate_greedy(
        model,
        input_ids,
        max_new_tokens=3,
        context_length=context_length,
    )

    assert output.shape == (1, 5)
