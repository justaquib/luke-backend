"""The Phase 7 token shapes: the question masked, the answer supervised, both
rendered by the same code the server asks with."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from app.services.language.training_materials import LESSON
from app.services.training import formats, splits
from app.services.training.dataset import PRETRAIN, SFT, Example

QUESTION = "What is rain?"
ANSWER = "Rain is water falling from a cloud."
PASSAGE = "Water the soil twice a week and keep it out of direct sun."


class WordTokenizer:
    """A tokenizer that gives each word one id, so a test can read a row back.

    It keeps its vocabulary between calls, which is what makes the asked half and
    the whole half share their leading ids here exactly as they do in a real
    byte-pair model.
    """

    chat_template = None
    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def ids(self, text: str) -> list[int]:
        out: list[int] = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 2
            out.append(self.vocab[word])
        return out

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.ids(text)}


class ChatTokenizer(WordTokenizer):
    """A tokenizer with a template, whose asked half is a generation prompt."""

    chat_template = "chatml"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [f"<|{item['role']}|> {item['content']} <|end|>" for item in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(parts)


class MergingTokenizer(WordTokenizer):
    """A tokenizer that fuses the answer marker with the first answer word.

    A byte-pair model does this across the boundary between the asked half and
    the whole half, and it is the reason the mask cannot always start exactly
    where the answer starts.
    """

    def ids(self, text: str) -> list[int]:
        words = text.split()
        merged: list[str] = []
        index = 0
        while index < len(words):
            if words[index] == "Answer:" and index + 1 < len(words):
                merged.append(f"Answer:+{words[index + 1]}")
                index += 2
            else:
                merged.append(words[index])
                index += 1
        return [self.vocab.setdefault(word, len(self.vocab) + 2) for word in merged]


def example(task: str = SFT, prompt: str = QUESTION, target: str = ANSWER) -> Example:
    return Example(task, f"{LESSON}:{prompt}", LESSON, prompt, target, splits.TRAIN)


class BatchEncodingLike(Mapping[str, list[int]]):
    """What a real HF tokenizer answers with: a mapping that is not a dict."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __getitem__(self, key: str) -> list[int]:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)


class MappingTokenizer(WordTokenizer):
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> BatchEncodingLike:
        return BatchEncodingLike(
            {"input_ids": self.ids(text), "attention_mask": [1] * len(self.ids(text))}
        )


class ListTokenizer(WordTokenizer):
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.ids(text)


def test_a_tokenizer_that_is_not_a_dict_still_gives_ids_and_not_names() -> None:
    for tokenizer in (MappingTokenizer(), ListTokenizer()):
        row = formats.sft_row(tokenizer, example())
        assert row is not None
        assert all(isinstance(token, int) for token in row.ids)
        assert row.ids == tuple(
            [*tokenizer.ids(formats.render(tokenizer, QUESTION, ANSWER, True)), 0]
        )


def test_the_question_is_masked_and_only_the_answer_carries_a_loss() -> None:
    tokenizer = WordTokenizer()
    row = formats.sft_row(tokenizer, example())
    assert row is not None
    asked = tokenizer.ids(formats.render(tokenizer, QUESTION, ANSWER, False))
    whole = tokenizer.ids(formats.render(tokenizer, QUESTION, ANSWER, True))
    assert len(asked) < len(whole)
    assert row.labels[: len(asked)] == (formats.IGNORE,) * len(asked)
    assert row.labels[len(asked) :] == row.ids[len(asked) :]
    assert row.supervised == len(row.ids) - len(asked)


def test_the_end_of_the_answer_is_part_of_the_loss() -> None:
    row = formats.sft_row(WordTokenizer(), example())
    assert row is not None
    assert row.ids[-1] == 0
    assert row.labels[-1] == 0


def test_a_model_with_a_template_is_rendered_through_its_own_template() -> None:
    tokenizer = ChatTokenizer()
    whole = formats.render(tokenizer, QUESTION, ANSWER, True)
    asked = formats.render(tokenizer, QUESTION, ANSWER, False)
    assert "<|user|>" in whole and "<|assistant|>" in whole
    assert whole.startswith(asked)
    row = formats.sft_row(tokenizer, example())
    assert row is not None
    assert row.supervised < len(row.ids)


def test_the_server_asks_in_the_shape_the_row_was_trained_in() -> None:
    for tokenizer in (WordTokenizer(), ChatTokenizer()):
        asked = formats.ask_text(tokenizer, QUESTION)
        assert formats.render(tokenizer, QUESTION, ANSWER, True).startswith(asked)


def test_an_answer_cut_off_by_the_token_budget_gives_no_row() -> None:
    assert formats.sft_row(WordTokenizer(), example(), max_tokens=4) is None


def test_a_passage_is_supervised_in_every_token() -> None:
    row = formats.pretrain_row(WordTokenizer(), example(PRETRAIN, "", PASSAGE))
    assert row is not None
    assert row.labels == row.ids
    assert formats.IGNORE not in row.labels
    assert row.supervised == len(row.ids)


def test_an_example_that_says_nothing_gives_no_row() -> None:
    tokenizer = WordTokenizer()
    assert formats.pretrain_row(tokenizer, example(PRETRAIN, "", "   ")) is None
    assert formats.sft_row(tokenizer, example(SFT, "", "   ")) is None


def test_the_boundary_token_between_the_two_halves_stays_supervised() -> None:
    tokenizer = MergingTokenizer()
    row = formats.sft_row(tokenizer, example())
    assert row is not None
    asked = tokenizer.ids(formats.render(tokenizer, QUESTION, ANSWER, False))
    assert row.labels[: len(asked) - 1] == (formats.IGNORE,) * (len(asked) - 1)
    assert row.labels[len(asked) - 1] != formats.IGNORE
    assert formats.IGNORE not in row.labels[len(asked) - 1 :]


def test_a_task_that_does_not_exist_is_named_rather_than_skipped() -> None:
    with pytest.raises(ValueError, match="Unknown training task"):
        formats.row(WordTokenizer(), example("distill"))


def test_rows_that_supervise_nothing_are_left_out() -> None:
    tokenizer = WordTokenizer()
    kept = example(prompt="Hi?", target="Rain.")
    dropped = example()
    built = formats.rows(tokenizer, [dropped, kept], max_tokens=5)
    assert [item.source for item in built] == [f"{LESSON}:Hi?"]
    assert all(item.supervised for item in built)


def test_the_padded_batch_keeps_the_padding_out_of_the_loss() -> None:
    tokenizer = WordTokenizer()
    rows = formats.rows(tokenizer, [example(), example(prompt="Hi?", target="A short one.")])
    batch = formats.pad(rows, pad_token_id=1)
    width = max(len(item.ids) for item in rows)
    assert all(len(ids) == width for ids in batch["input_ids"])
    assert all(len(labels) == width for labels in batch["labels"])
    assert all(len(mask) == width for mask in batch["attention_mask"])
    for position, row in enumerate(rows):
        trailing = width - len(row.ids)
        assert batch["attention_mask"][position][len(row.ids) :] == [0] * trailing
        assert batch["labels"][position][len(row.ids) :] == [formats.IGNORE] * trailing
        assert batch["input_ids"][position][len(row.ids) :] == [1] * trailing
