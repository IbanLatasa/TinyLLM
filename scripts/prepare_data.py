import json
from pathlib import Path

from datasets import Dataset, load_dataset


def save_split(
    dataset_split: Dataset,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stories_saved = 0

    with output_path.open("w", encoding="utf-8") as file:
        for sample in dataset_split:
            text = sample["text"]

            if not text.strip():
                continue

            record = {
                "text": text,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            stories_saved += 1

    return stories_saved


def main() -> None:
    dataset = load_dataset("roneneldan/TinyStories")

    train_count = save_split(
        dataset["train"],
        Path("data/processed/train.jsonl"),
    )

    validation_count = save_split(
        dataset["validation"],
        Path("data/processed/validation.jsonl"),
    )

    print(f"Train stories: {train_count}")
    print(f"Validation stories: {validation_count}")


if __name__ == "__main__":
    main()
