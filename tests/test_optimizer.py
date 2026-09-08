import torch
from torch import nn

from tinyllm.training.optimizer import create_optimizer


def test_optimizer_updates_model_parameters() -> None:
    model = nn.Linear(4, 2)

    optimizer = create_optimizer(
        model,
        learning_rate=1e-2,
    )

    x = torch.randn(3, 4)
    target = torch.randn(3, 2)

    initial_weight = model.weight.detach().clone()

    output = model(x)

    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.equal(
        initial_weight,
        model.weight,
    )
