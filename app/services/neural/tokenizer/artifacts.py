"""The content-addressed artifacts a scratch tokenizer is read back from.

An artifact is named by the SHA-256 of its ``tokenizer.json``, so a changed
vocabulary or a changed merge list is a different artifact and can never be
mistaken for the old one. The files are written in a folder of their own and
moved into place in one move, and are verified by hash before use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from app.services.brain_paths import brain_dir
from app.services.neural.tokenizer.base import (
    BPE_KIND,
    META_FILE,
    TOKENIZER_FILE,
    WORD_KIND,
    CorruptArtifactError,
    MissingArtifactError,
    Tokenizer,
    VocabularyMismatchError,
)
from app.services.neural.tokenizer.bpe import BpeTokenizer
from app.services.neural.tokenizer.word import WordTokenizer

LOGGER = logging.getLogger(__name__)

SCHEMA = 1
FORMAT = "luke-tokenizer"
# The scratch tokenizers are trained by this app, with this licence.
LICENCE = "Apache-2.0"


def root() -> Path:
    """The folder every recorded artifact is kept in."""
    return brain_dir() / "tokenizers"


def folder(artifact_id: str) -> Path:
    return root() / artifact_id


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _implementation(tokenizer: Tokenizer) -> tuple[str, str]:
    if tokenizer.kind == WORD_KIND:
        return "luke-word", "1"
    from tokenizers import __version__

    return "tokenizers", str(__version__)


def record(tokenizer: Tokenizer, corpus: dict[str, Any] | None = None) -> dict[str, Any]:
    """Save a tokenizer and keep the record its checkpoint points at.

    The artifact id is the hash of the saved file, so the same vocabulary is
    the same artifact and a different one can never replace it. A word
    tokenizer needs no artifact: its vocabulary travels in the checkpoint.
    """
    if tokenizer.kind == WORD_KIND:
        return {
            "kind": tokenizer.kind,
            "id": tokenizer.digest(),
            "vocab_size": tokenizer.vocab_size,
            "special_ids": tokenizer.special_ids,
        }
    root().mkdir(parents=True, exist_ok=True)
    temporary = root() / f".tmp-{uuid.uuid4().hex}"
    try:
        tokenizer.save(temporary)
        data = (temporary / TOKENIZER_FILE).read_bytes()
        artifact_id = sha256(data)
        implementation, version = _implementation(tokenizer)
        meta = {
            "format": FORMAT,
            "schema": SCHEMA,
            "kind": tokenizer.kind,
            "id": artifact_id,
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": list(tokenizer.special_ids),
            "special_ids": tokenizer.special_ids,
            "implementation": implementation,
            "implementation_version": version,
            "licence": LICENCE,
            "created": time.time(),
            "corpus": dict(corpus or {}),
            **tokenizer.describe(),
        }
        (temporary / META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        final = folder(artifact_id)
        if final.is_dir():
            _verify_folder(final, artifact_id)
        else:
            os.replace(temporary, final)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        "kind": tokenizer.kind,
        "id": artifact_id,
        "vocab_size": tokenizer.vocab_size,
        "special_ids": tokenizer.special_ids,
    }


def _verify_folder(path: Path, artifact_id: str) -> None:
    tokenizer_file = path / TOKENIZER_FILE
    if not tokenizer_file.is_file():
        raise MissingArtifactError(f"the tokenizer artifact {artifact_id} has no tokenizer file")
    if sha256(tokenizer_file.read_bytes()) != artifact_id:
        raise CorruptArtifactError(f"the tokenizer artifact {artifact_id} does not match its hash")


def meta(artifact_id: str) -> dict[str, Any]:
    """The recorded shape of an artifact."""
    path = folder(artifact_id) / META_FILE
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MissingArtifactError(
            f"the tokenizer artifact {artifact_id} has no readable meta"
        ) from error
    if not isinstance(stored, dict):
        raise CorruptArtifactError(f"the tokenizer artifact {artifact_id} has no meta object")
    return stored


def load(recorded: dict[str, Any], vocabulary: list[str] | None = None) -> Tokenizer:
    """Read the tokenizer a checkpoint names, and refuse a mismatch.

    The word adapter reads the vocabulary the checkpoint carries; a BPE
    artifact is read from its folder, verified by hash and by size. Nothing is
    remapped, truncated or silently replaced when the two disagree.
    """
    kind = str(recorded.get("kind") or "")
    size = int(recorded.get("vocab_size") or 0)
    if kind == WORD_KIND:
        if vocabulary is None:
            raise VocabularyMismatchError("a word checkpoint must carry its vocabulary")
        tokenizer = WordTokenizer(vocabulary)
    elif kind == BPE_KIND:
        tokenizer = _load_bpe(str(recorded.get("id") or ""))
    else:
        raise CorruptArtifactError(f"the checkpoint names an unknown tokenizer kind '{kind}'")
    if tokenizer.vocab_size != size:
        raise VocabularyMismatchError(
            f"the checkpoint expects {size} pieces but the tokenizer holds {tokenizer.vocab_size}"
        )
    return tokenizer


def _load_bpe(artifact_id: str) -> BpeTokenizer:
    if not artifact_id:
        raise CorruptArtifactError("the checkpoint names a BPE tokenizer with no artifact id")
    path = folder(artifact_id)
    if not path.is_dir():
        raise MissingArtifactError(f"the tokenizer artifact {artifact_id} is not on disk")
    _verify_folder(path, artifact_id)
    stored = meta(artifact_id)
    if str(stored.get("kind")) != BPE_KIND:
        raise CorruptArtifactError(f"the artifact {artifact_id} is not a BPE tokenizer")
    return BpeTokenizer.load(path)


def verify(recorded: dict[str, Any], vocabulary_size: int) -> None:
    """Check the record against the weights before the weights are built."""
    size = int(recorded.get("vocab_size") or 0)
    if size != int(vocabulary_size):
        raise VocabularyMismatchError(
            f"the weights hold {vocabulary_size} rows but the tokenizer record says {size}"
        )
    if str(recorded.get("kind") or "") == BPE_KIND:
        for name, token_id in dict(recorded.get("special_ids") or {}).items():
            if not 0 <= int(token_id) < size:
                raise VocabularyMismatchError(
                    f"the special mark {name} has id {token_id}, outside a vocabulary of {size}"
                )


def describe(recorded: dict[str, Any]) -> dict[str, Any]:
    """What the page may show about a checkpoint's tokenizer, without loading it."""
    from app.services.neural.tokenizer import spec

    kind = str(recorded.get("kind") or "")
    artifact_id = str(recorded.get("id") or "")
    row: dict[str, Any] = {
        "kind": kind or "not recorded",
        "default_kind": spec.default_kind(),
        "id": artifact_id,
        "short_id": artifact_id[:12],
        "vocab_size": int(recorded.get("vocab_size") or 0),
        "special_ids": dict(recorded.get("special_ids") or {}),
        "artifact_bytes": 0,
        "present": kind == WORD_KIND,
        "error": None,
    }
    if kind == BPE_KIND and artifact_id:
        path = folder(artifact_id)
        tokenizer_file = path / TOKENIZER_FILE
        try:
            _verify_folder(path, artifact_id)
            row["artifact_bytes"] = tokenizer_file.stat().st_size
            row["present"] = True
        except (MissingArtifactError, CorruptArtifactError, OSError) as error:
            row["error"] = str(error)
        try:
            stored = meta(artifact_id)
        except (MissingArtifactError, CorruptArtifactError):
            stored = {}
        corpus = dict(stored.get("corpus") or {})
        characters = int(corpus.get("characters") or 0)
        pieces = int(corpus.get("tokens") or 0)
        row["corpus"] = corpus
        row["tokens_per_character"] = round(pieces / characters, 4) if characters else None
        row["created"] = stored.get("created")
        row["licence"] = stored.get("licence")
        row["implementation"] = stored.get("implementation")
    elif kind not in {WORD_KIND, ""}:
        row["error"] = f"unknown tokenizer kind '{kind}'"
    return row


def artifact_bytes(recorded: dict[str, Any]) -> int:
    """The size of the tokenizer file a checkpoint points at."""
    artifact_id = str(recorded.get("id") or "")
    if str(recorded.get("kind") or "") != BPE_KIND or not artifact_id:
        return 0
    try:
        return (folder(artifact_id) / TOKENIZER_FILE).stat().st_size
    except OSError:
        return 0
