import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tinyllm.data.dataset import LanguageModelDataset
from tinyllm.model.gpt import GPT
from tinyllm.tokenizer.bpe import TinyLLMTokenizer
from tinyllm.training.loss import LanguageModelLoss
from tinyllm.training.metrics import perplexity
from tinyllm.training.trainer import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyLLM checkpoints")

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="directory containing step checkpoints",
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
        help="path to tokenizer",
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
        help="path to validation data",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="validation batch size"
    )

    return parser.parse_args()


def load_stories(path: Path, tokenizer: TinyLLMTokenizer) -> list[list[int]]:
    stories: list[list[int]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            text = record["text"]

            if not text.strip():
                continue

            stories.append(tokenizer.encode(text))

    return stories


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GPT, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["model_config"]

    model = GPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        context_length=config["context_length"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = sorted(args.checkpoint_dir.glob("step_*.pt"))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No step checkpoint found in {args.checkpoint_dir}")

    tokenizer = TinyLLMTokenizer.from_file(args.tokenizer)

    validation_stories = load_stories(args.validation_data, tokenizer)

    bos_token_id = tokenizer.token_to_id("<bos>")
    eos_token_id = tokenizer.token_to_id("<eos>")
    pad_token_id = tokenizer.token_to_id("<pad>")

    first_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")

    context_length = first_checkpoint["model_config"]["context_length"]

    validation_dataset = LanguageModelDataset(
        stories=validation_stories,
        context_length=context_length,
        eos_id=eos_token_id,
        bos_id=bos_token_id,
        pad_id=pad_token_id,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    loss_fn = LanguageModelLoss(pad_token_id=pad_token_id)

    print()
    print("TinyLLM checkpoint evaluation")
    print("------------------------------")
    print(f"Device: {device}")
    print(f"Checkpoints: {len(checkpoint_paths)}")
    print(f"Validation stories: {len(validation_stories)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print()

    results = []

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating {checkpoint_path.name}...")

        model, checkpoint = load_model(checkpoint_path, device)

        validation_loss = validate(
            model=model,
            dataloader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            pad_token_id=pad_token_id,
        )

        validation_perplexity = perplexity(validation_loss)

        step = checkpoint["step"]

        results.append(
            {"step": step, "loss": validation_loss, "perplexity": validation_perplexity}
        )

        print(
            f"Step: {step} "
            f"| val_loss={validation_loss} "
            f"| val_ppl={validation_perplexity}"
        )

        del model
        del checkpoint

        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_result = min(results, key=lambda result: result["loss"])

    print()
    print("Results")
    print("--------")
    print(f"{'Step'} {'Val Loss'} {'Val PPL'}")
    for result in results:
        marker = "<-- best" if result["step"] == best_result["step"] else ""
        print(f"{result['step']} {result['loss']} {result['perplexity']} {marker}")
    print()
    print(f"Best checkpoint: step_{best_result['step']}.pt")
    print(f"Best validation loss: {best_result['loss']}")
    print(f"Best validation perplexity: {best_result['perplexity']}")


if __name__ == "__main__":
    main()
