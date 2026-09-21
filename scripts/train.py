import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinyllm.data.dataset import LanguageModelDataset
from tinyllm.model.gpt import GPT
from tinyllm.tokenizer.bpe import TinyLLMTokenizer
from tinyllm.training.checkpoint import load_checkpoint, save_checkpoint
from tinyllm.training.loss import LanguageModelLoss
from tinyllm.training.metrics import perplexity
from tinyllm.training.optimizer import create_optimizer
from tinyllm.training.trainer import train_epoch, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyLLM")

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint from which to resume training",
    )

    return parser.parse_args()


def load_stories(
    path: Path, tokenizer: TinyLLMTokenizer, max_stories: int | None = None
) -> list[list[int]]:
    stories: list[list[int]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            text = record["text"]

            if not text.strip():
                continue

            stories.append(tokenizer.encode(text=text))

            if max_stories is not None and len(stories) >= max_stories:
                break

    return stories


def main() -> None:
    args = parse_args()

    tokenizer_path = Path("artifacts/tokenizer/tokenizer.json")
    train_path = Path("data/processed/train.jsonl")
    validation_path = Path("data/processed/validation.jsonl")
    checkpoint_dir = Path("artifacts/checkpoints")

    # Model conf
    context_length = 128
    embedding_dim = 128
    num_heads = 4
    num_layers = 4

    # Training conf
    batch_size = 16
    learning_rate = 3e-4

    max_steps = 500_000
    log_every = 1000
    checkpoint_every = 50_000

    use_amp = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TinyLLMTokenizer.from_file(tokenizer_path)

    train_stories = load_stories(train_path, tokenizer)

    validation_stories = load_stories(validation_path, tokenizer)

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    pad_token_id = tokenizer.token_to_id("<pad>")

    vocab_size = tokenizer.vocab_size()

    train_dataset = LanguageModelDataset(
        stories=train_stories,
        context_length=context_length,
        bos_id=bos_token_id,
        eos_id=eos_token_id,
        pad_id=pad_token_id,
    )

    validation_dataset = LanguageModelDataset(
        stories=validation_stories,
        context_length=context_length,
        eos_id=eos_token_id,
        bos_id=bos_token_id,
        pad_id=pad_token_id,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        # persistent_workers=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        # persistent_workers=True,
    )

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    loss_fn = LanguageModelLoss(pad_token_id=pad_token_id)

    optimizer = create_optimizer(model, learning_rate=learning_rate)

    model_config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "context_length": context_length,
        "num_heads": num_heads,
        "num_layers": num_layers,
    }

    start_step = 0
    scaler_state_dict = None

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

        checkpoint = load_checkpoint(
            path=args.resume, model=model, optimizer=optimizer, device=device
        )

        checkpoint_config = checkpoint["model_config"]

        if checkpoint_config != model_config:
            raise ValueError(
                "Checkpoint model configuration "
                "does not match current model configuration"
            )

        start_step = checkpoint["step"]

        scaler_state_dict = checkpoint.get("scaler_state_dict")

        print(f"Resuming from: {args.resume}")
        print(f"Startint at step: {start_step}")

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    print()
    print("TinyLLM Training")
    print("-----------------")
    print(f"Device: {device}")
    print(f"AMP: {use_amp and device.type == 'cuda'}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training stories: {len(train_stories)}")
    print(f"Validation stories: {len(validation_stories)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Model parameters: {num_parameters:,}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps:,}")
    print(f"Checkpoint every: {checkpoint_every}")
    print()

    def save_intermediate_checkpoint(
        step: int, train_loss: float, tokens_per_second: float, scaler_state: dict
    ) -> None:
        path = checkpoint_dir / f"step_{step:06d}.pt"

        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            step=step,
            train_loss=train_loss,
            validation_loss=None,
            model_config=model_config,
            scaler_state_dict=scaler_state,
        )

        print(f"Checkpoint saved: {path} | tokens/s={tokens_per_second:,.0f}")

    train_loss, tokens_per_second, final_scaler_state = train_epoch(
        model=model,
        dataloader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        max_steps=max_steps,
        pad_token=pad_token_id,
        use_amp=use_amp,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        start_step=start_step,
        scaler_state_dict=scaler_state_dict,
        on_checkpoint=save_intermediate_checkpoint,
    )

    print()
    print("Running validation...")

    validation_loss = validate(
        model=model,
        dataloader=validation_loader,
        loss_fn=loss_fn,
        device=device,
        pad_token_id=pad_token_id,
    )

    train_ppl = perplexity(train_loss)

    validation_ppl = perplexity(validation_loss)

    print()
    print("Training complete")
    print("------------------")
    print(f"Steps: {max_steps:,}")
    print(f"Train loss: {train_loss}")
    print(f"Train ppl: {train_ppl}")
    print(f"Validation loss: {validation_loss}")
    print(f"Validation ppl: {validation_ppl}")
    print(f"Throughput: {tokens_per_second:,.0f} tokens/s")

    final_checkpoint_path = checkpoint_dir / "latest.pt"

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=max_steps,
        train_loss=train_loss,
        validation_loss=validation_loss,
        model_config=model_config,
    )

    print()
    print(f"Final checkpoint saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
