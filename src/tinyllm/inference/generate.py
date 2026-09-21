import torch
from torch import Tensor, nn


@torch.no_grad()
def generate_greedy(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    context_length: int,
    eos_token_id: int | None = None,
) -> Tensor:
    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length]

        logits = model(model_input)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return generated


@torch.no_grad()
def generate_with_temperature(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length]

        logits = model(model_input)

        next_token_logits = logits[:, -1, :]

        scaled_logits = next_token_logits / temperature

        probabilities = torch.softmax(scaled_logits, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return generated


@torch.no_grad()
def generate_with_top_k(
    model: nn.Module,
    input_ids: Tensor,
    max_new_tokens: int,
    context_length: int,
    top_k: int,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length]

        logits = model(model_input)

        next_token_logits = logits[:, -1, :]

        vocab_size = next_token_logits.size(-1)

        if top_k > vocab_size:
            raise ValueError("top_k cannot be greater than vocab_size")

        scaled_logits = next_token_logits / temperature

        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)

        probabilities = torch.softmax(top_k_logits, dim=-1)

        sampled_index = torch.multinomial(probabilities, num_samples=1)

        next_token = torch.gather(top_k_indices, dim=-1, index=sampled_index)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return generated
