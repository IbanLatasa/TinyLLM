from pathlib import Path

from tinyllm.tokenizer.bpe import train_tokenizer

TRAIN_PATH = Path("data/processed/train.txt")
OUTPUT_PATH = Path("artifacts/tokenizer/tokenizer.json")


def main() -> None:
    train_tokenizer(input_path=TRAIN_PATH, output_path=OUTPUT_PATH, vocab_size=8_000)

    print(f"Tokenizer saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
