from pathlib import Path

from tinyllm.tokenizer.bpe import TinyLLMTokenizer, train_tokenizer


def build_test_tokenizer(tmp_path: Path) -> TinyLLMTokenizer:
    corpus_path = tmp_path / "train.txt"
    tokenizer_path = tmp_path / "tokenizer.json"

    corpus_path.write_text(
        "Once upon a time.\nThere was a little dragon.\nThe dragon was happy.\n",
        encoding="utf-8",
    )

    train_tokenizer(input_path=corpus_path, output_path=tokenizer_path, vocab_size=100)

    return TinyLLMTokenizer.from_file(tokenizer_path)


def test_encode_decode_round_trip(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)

    text = "Once upon a time."

    token_ids = tokenizer.encode(text=text)
    decoded = tokenizer.decode(token_ids=token_ids)

    assert decoded == text


def test_encode_returns_integers(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)

    token_ids = tokenizer.encode("Once upon a time.")

    assert all(isinstance(token_id, int) for token_id in token_ids)


def test_empty_string(tmp_path: Path) -> None:
    tokenizer = build_test_tokenizer(tmp_path)

    token_ids = tokenizer.encode("")
    decoded = tokenizer.decode(token_ids=token_ids)

    assert token_ids == []
    assert decoded == ""


def test_tokenizer_can_be_reloaded(tmp_path: Path) -> None:
    corpus_path = tmp_path / "train.txt"
    tokenizer_path = tmp_path / "tokenizer.json"

    corpus_path.write_text(
        "Once upon a time.\nThere was a little dragon.\n",
        encoding="utf-8",
    )

    train_tokenizer(
        input_path=corpus_path,
        output_path=tokenizer_path,
        vocab_size=100,
    )

    tokenizer_1 = TinyLLMTokenizer.from_file(tokenizer_path)
    token_ids_before = tokenizer_1.encode("Once upon a time.")

    tokenizer_2 = TinyLLMTokenizer.from_file(tokenizer_path)
    token_ids_after = tokenizer_2.encode("Once upon a time.")

    assert token_ids_before == token_ids_after
    assert tokenizer_2.decode(token_ids_after) == "Once upon a time."
