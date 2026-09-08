from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    "<unk>",
    "<bos>",
    "<eos>",
    "<pad>",
]


class TinyLLMTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: Path) -> "TinyLLMTokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)

        if token_id is None:
            raise ValueError(f"Unknown token: {token}")

        return token_id

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def train_tokenizer(
    input_path: Path, output_path: Path, vocab_size: int = 8_000
) -> None:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)

    tokenizer.train(files=[str(input_path)], trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_path))
