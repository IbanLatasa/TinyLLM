from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinyllm.inference.generate import (
    generate_greedy,
    generate_with_temperature,
    generate_with_top_k,
)
from tinyllm.tokenizer.bpe import TinyLLMTokenizer
from tinyllm.training.checkpoint import load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with TinyLLM")

    parser.add_argument(
        "--prompt", type=str, required=True, help="Input text used to start generation"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/latest.pt"),
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizer/tokenizer.json"),
        help="Path to tokenizer",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="maximum number of tokens to generate",
    )

    parser.add_argument(
        "--strategy",
        choices=["greedy", "temperature", "top-k"],
        default="greedy",
        help="generation strategy",
    )

    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )

    parser.add_argument(
        "--top-k", type=int, default=50, help="Number of candidates for top-k sampling"
    )

    parser.add_argument(
        "--seed", type=int, default=33, help="Random seed for reproducible generation"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TinyLLMTokenizer.from_file(args.tokenizer)

    model, checkpoint = load_model_checkpoint(path=args.checkpoint, device=device)

    model_config = checkpoint["model_config"]

    token_ids = tokenizer.encode(args.prompt)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    eos_token_id = tokenizer.token_to_id("<eos>")

    if args.strategy == "greedy":
        output_ids = generate_greedy(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            context_length=model_config["context_length"],
            eos_token_id=eos_token_id,
        )

    elif args.strategy == "temperature":
        output_ids = generate_with_temperature(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            context_length=model_config["context_length"],
            temperature=args.temperature,
            eos_token_id=eos_token_id,
        )

    else:
        output_ids = generate_with_top_k(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            context_length=model_config["context_length"],
            top_k=args.top_k,
            temperature=args.temperature,
            eos_token_id=eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0].tolist())

    print(generated_text)


if __name__ == "__main__":
    main()
