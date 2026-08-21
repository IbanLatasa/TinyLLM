from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<unk>",
    "<eos>",
    "<pad>",
]


def train_tokenizer(
    input_path: Path, output_path: Path, vocab_size: int = 8_000
) -> None:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    tokenizer.train(files=[str(input_path)], trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_path))
