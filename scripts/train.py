import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinyllm.data.dataset import LanguageModelDataset
from tinyllm.model.gpt import GPT
from tinyllm.tokenizer.bpe import TinyLLMTokenizer
from tinyllm.training.checkpoint import save_checkpoint
from tinyllm.training.loss import LanguageModelLoss
from tinyllm.training.metrics import perplexity
from tinyllm.training.optimizer import create_optimizer
from tinyllm.training.trainer import train_epoch, validate


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
    tokenizer_path = Path("artifacts/tokenizer/tokenizer.json")
    train_path = Path("data/processed/train.jsonl")
    validation_path = Path("data/processed/validation.jsonl")

    context_length = 128
    embedding_dim = 128
    num_heads = 4
    num_layers = 4
    batch_size = 16
    learning_rate = 3e-4
    num_epochs = 5

    tokenizer = TinyLLMTokenizer.from_file(tokenizer_path)

    train_stories = load_stories(train_path, tokenizer, max_stories=10000)

    validation_stories = load_stories(validation_path, tokenizer, max_stories=10000)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    loss_fn = LanguageModelLoss(pad_token_id=pad_token_id)

    optimizer = create_optimizer(model, learning_rate=learning_rate)

    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training stories: {len(train_stories)}")
    print(f"Validation stories: {len(validation_stories)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    num_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model parameters: {num_parameters:,}")

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = validate(
            model=model, dataloader=validation_loader, loss_fn=loss_fn, device=device
        )

        train_ppl = perplexity(train_loss)
        validation_ppl = perplexity(validation_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| train_ppl={train_ppl:.2f} "
            f"| val_loss={validation_loss:.4f} "
            f"| val_ppl={validation_ppl:.2f}"
        )

        save_checkpoint(
            path=Path("artifacts/checkpoints/latest.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
        )


if __name__ == "__main__":
    main()
