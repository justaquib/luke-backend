"""The one vocabulary.

The transformer's token list is the only live token→id mapping in the app.
Anything that reads a vocabulary reads this one, through :func:`canonical`.

The mappings that came before it are kept as read-only artefacts in
``app/legacy``. They are frozen for the record: they are never used at
inference, and they are never loaded by the running app. The startup guard
(:func:`assert_single_vocabulary`) fails loudly if one of them reaches a
tensor bound for the transformer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

CANONICAL_OWNER = "transformer"
LEGACY_DIRECTORY = Path(__file__).resolve().parent.parent / "legacy"

LOGGER = logging.getLogger(__name__)


class VocabularyError(RuntimeError):
    """Raised when a retired vocabulary reaches a live path."""


class Vocabulary:
    """A token list and its ids.

    Only one of these is live: the transformer's. A retired one can still be
    read, for the record, but it cannot turn text into ids, because ids from
    an old mapping would silently mean different words in the new model.
    """

    def __init__(self, owner: str, tokens: Iterable[str], retired: bool = False) -> None:
        self.owner = owner
        self.tokens = tuple(str(token) for token in tokens)
        self.retired = retired
        self._index = {token: position for position, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def index(self) -> dict[str, int]:
        return dict(self._index)

    def __contains__(self, token: object) -> bool:
        return token in self._index

    def id_of(self, token: str) -> int | None:
        """The id of a token, or None when the vocabulary does not hold it."""
        return self._index.get(token)

    def token_of(self, token_id: int) -> str | None:
        """The token of an id, or None when the id is outside the vocabulary."""
        if 0 <= token_id < len(self.tokens):
            return self.tokens[token_id]
        return None

    def encode(self, text: str) -> list[int]:
        """Turn text into ids.

        A retired vocabulary refuses to do this: ids from an old mapping would
        be read as different words by the model that answers.
        """
        if self.retired:
            raise VocabularyError(
                f"the '{self.owner}' vocabulary is retired and must not produce ids"
            )
        from app.services.neural.tokens import tokenize

        unknown = self._index.get("<unk>")
        return [
            self._index.get(token, unknown if unknown is not None else 0)
            for token in tokenize(text)
        ]

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(token for token in (self.token_of(int(i)) for i in ids) if token)


def canonical(tenant_id: int | None = None) -> Vocabulary:
    """The transformer's vocabulary: the only live token→id mapping."""
    from app.services import brain

    writer = brain.load_writer(tenant_id)
    tokens = tuple(writer.vocabulary) if writer is not None else ()
    return Vocabulary(CANONICAL_OWNER, tokens)


def retired(name: str) -> Vocabulary:
    """A frozen mapping from ``app/legacy``, for the record only."""
    path = LEGACY_DIRECTORY / name
    if not path.is_file():
        raise VocabularyError(f"the retired vocabulary '{name}' is not in {LEGACY_DIRECTORY}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    return Vocabulary(stored.get("owner", "retired"), stored.get("tokens", []), retired=True)


def retired_size(name: str) -> int:
    """How many words a frozen artefact holds."""
    try:
        return len(retired(name).tokens)
    except (OSError, ValueError, VocabularyError) as error:
        LOGGER.warning("the retired vocabulary '%s' could not be read: %s", name, error)
        return 0


def retired_names() -> list[str]:
    if not LEGACY_DIRECTORY.is_dir():
        return []
    return sorted(path.name for path in LEGACY_DIRECTORY.glob("*.json"))


def guard_tensor_ids(
    ids: Iterable[int], source: str, vocabulary: Vocabulary | None = None
) -> list[int]:
    """Check ids before they are bound for the transformer.

    An id that the canonical vocabulary does not hold would be read as
    whatever word sits at that position, so it is refused here instead of
    quietly changing the answer.
    """
    vocab = vocabulary if vocabulary is not None else canonical()
    if vocab.retired:
        raise VocabularyError(f"{source} used the retired '{vocab.owner}' vocabulary")
    checked: list[int] = []
    for token_id in ids:
        value = int(token_id)
        if value < 0 or value >= len(vocab):
            raise VocabularyError(
                f"{source} sent id {value}, which is outside "
                f"the canonical vocabulary of {len(vocab)}"
            )
        checked.append(value)
    return checked


def _identity_problems(vocab: Vocabulary) -> list[str]:
    """The live mapping must be the canonical one, and hold no token twice."""
    problems: list[str] = []
    if vocab.owner != CANONICAL_OWNER:
        problems.append(f"the live vocabulary is '{vocab.owner}', not '{CANONICAL_OWNER}'")
    if len(vocab.index) != len(vocab):
        problems.append("the canonical vocabulary holds the same token more than once")
    return problems


def _produces_ids(vocab: Vocabulary) -> bool:
    """Whether a mapping can still turn text into ids."""
    try:
        vocab.encode("the cow is an animal")
    except VocabularyError:
        return False
    return True


def _retired_problems(vocab: Vocabulary) -> list[str]:
    """A retired mapping must refuse ids, and must not share the live ids."""
    problems: list[str] = []
    for name in retired_names():
        try:
            old = retired(name)
        except (OSError, ValueError, VocabularyError) as error:
            problems.append(f"the retired vocabulary '{name}' could not be read: {error}")
            continue
        if _produces_ids(old):
            problems.append(f"the retired vocabulary '{name}' can still produce ids")
        # The old ids overlap the new ones, so a mixed id would look valid.
        # The guard is what stops it, and the guard has to be reached.
        shared = [token for token in old.tokens if token in vocab]
        if shared:
            different = [token for token in shared if old.id_of(token) != vocab.id_of(token)]
            if not different:
                problems.append(
                    f"the retired vocabulary '{name}' has the same ids as the canonical one"
                )
    return problems


def _boundary_problems() -> list[str]:
    """An id from a retired mapping must be refused at the tensor boundary."""
    problems: list[str] = []
    for name in retired_names():
        try:
            old = retired(name)
        except (OSError, ValueError, VocabularyError):
            continue
        legacy_ids = list(range(len(old.tokens)))
        try:
            guard_tensor_ids(legacy_ids, f"the retired mapping '{name}'", old)
        except VocabularyError:
            continue
        problems.append(f"the retired mapping '{name}' was accepted at the tensor boundary")
    return problems


def assert_single_vocabulary(tenant_id: int | None = None) -> dict:
    """Check that only the transformer's mapping can reach the answering model.

    This runs at startup, so a mistake stops the app instead of quietly
    answering with the wrong words.
    """
    vocab = canonical(tenant_id)
    problems = _identity_problems(vocab)
    problems += _retired_problems(vocab)

    # The answering model must be able to read its own ids.
    if len(vocab):
        try:
            guard_tensor_ids(vocab.encode("the cow is an animal"), "startup check")
        except VocabularyError as error:
            problems.append(f"the canonical vocabulary failed its own check: {error}")

    # An id from a retired mapping must be refused at the tensor boundary.
    # The two id spaces overlap, so a number alone cannot be told apart: the
    # guard refuses the retired mapping itself, and that is what stops it.
    problems += _boundary_problems()

    if problems:
        raise VocabularyError(
            "the app must run on one vocabulary, the transformer's:\n- " + "\n- ".join(problems)
        )
    return {
        "owner": vocab.owner,
        "size": len(vocab),
        "trained": len(vocab) > 0,
        "retired": retired_names(),
    }
