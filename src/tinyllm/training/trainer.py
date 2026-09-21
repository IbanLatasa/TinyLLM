import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

CheckpointCallback = Callable[[int, float, float, dict[str, Any]], None]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_steps: int,
    pad_token: int,
    use_amp: bool = False,
    log_every: int = 100,
    checkpoint_every: int | None = None,
    start_step: int = 0,
    scaler_state_dict: dict[str, Any] | None = None,
    on_checkpoint: CheckpointCallback | None = None,
) -> tuple[float, float, dict[str, Any]]:
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than 0")
    if start_step < 0:
        raise ValueError("start_step cannot be negative")

    if start_step >= max_steps:
        raise ValueError("start_step must be smaller than max_steps")

    if log_every <= 0:
        raise ValueError("log_every must be greater than 0")

    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be greater than 0")

    model.train()

    total_loss = 0.0
    total_tokens = 0
    steps_this_run = 0

    global_step = start_step

    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    start_time = time.perf_counter()

    while global_step < max_steps:
        for x, y in dataloader:
            if global_step >= max_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(x)

                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

            batch_tokens = (y != pad_token).sum().item()

            total_tokens += batch_tokens

            global_step += 1
            steps_this_run += 1

            if global_step % log_every == 0:
                elapsed = time.perf_counter() - start_time

                average_loss = total_loss / steps_this_run

                tokens_per_second = total_tokens / elapsed

                print(
                    f"Step: {global_step}/{max_steps} "
                    f"| loss={average_loss} "
                    f"| tokens/s={tokens_per_second}"
                )

            if (
                checkpoint_every is not None
                and on_checkpoint is not None
                and global_step % checkpoint_every == 0
            ):
                elapsed = time.perf_counter() - start_time

                average_loss = total_loss / steps_this_run

                tokens_per_second = total_tokens / elapsed

                on_checkpoint(
                    global_step,
                    average_loss,
                    tokens_per_second,
                    scaler.state_dict(),
                )

    elapsed = time.perf_counter() - start_time

    average_loss = total_loss / steps_this_run

    tokens_per_second = total_tokens / elapsed

    return average_loss, tokens_per_second, scaler.state_dict()


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    pad_token_id: int,
) -> float:
    model.eval()

    total_loss = 0.0
    total_valid_tokens = 0

    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            valid_tokens = (y != pad_token_id).sum().item()

            total_loss += loss.item() * valid_tokens

            total_valid_tokens += valid_tokens

    return total_loss / total_valid_tokens
