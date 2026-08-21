from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path("data/processed")


def save_split(dataset_split, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_stories = 0

    with output_path.open("w", encoding="utf-8") as file:
        for story in dataset_split:
            text = story["text"].strip()

            if text:
                file.write(text)
                saved_stories += 1

    return saved_stories


def main() -> None:
    ds = load_dataset("roneneldan/TinyStories")

    train_count = save_split(ds["train"], OUTPUT_DIR / "train.txt")
    val_count = save_split(ds["validation"], OUTPUT_DIR / "validation.txt")

    print(f"Saved train split: {train_count:,} stories")
    print(f"Saved validation split: {val_count:,} stories")


if __name__ == "__main__":
    main()
