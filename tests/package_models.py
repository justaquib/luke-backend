"""Tiny readable checkpoints and tokenizers used only by the package tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models

from app.services.foundation import identity
from app.services.neural import checkpoint
from app.services.neural.model import TinyTransformer
from app.services.neural.tokenizer import artifacts
from app.services.neural.tokenizer.bpe import BpeTokenizer
from app.services.neural.tokenizer.word import WordTokenizer

TEXT = "The moon reflects sunlight. A drum is hit with a stick."


def write_scratch(root: Path, workspace: int, kind: str = "bpe") -> dict:
    """Save a schema-three word checkpoint without training or downloading."""
    tokenizer = (
        BpeTokenizer.train([TEXT] * 4, 280) if kind == "bpe" else WordTokenizer.build([TEXT])
    )
    record = artifacts.record(tokenizer)
    config = {
        "d_model": 8,
        "layers": 1,
        "heads": 2,
        "context": 8,
        "pad_id": tokenizer.special_ids["<pad>"],
        "tokenizer_kind": kind,
    }
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = TinyTransformer(tokenizer.vocab_size, config)
    checkpoint.save(
        root / f"words-{workspace}.pt",
        checkpoint.Checkpoint(
            task="words",
            model=model,
            vocabulary=tokenizer.vocabulary(),
            config=config,
            metrics={},
            tokenizer=record,
        ),
    )
    return record


def write_base(offset: int = 0) -> Path:
    """Build a real safetensors file, tokenizer and matching measured identity."""
    folder = identity.install_dir()
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / identity.WEIGHT_FILE
    tensor = (torch.arange(6, dtype=torch.float32) + offset).reshape(2, 3)
    save_file({"w": tensor}, str(weights))
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "moon": 1}, unk_token="[UNK]")  # noqa: S106 -- a token
    )
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (folder / "config.json").write_text(
        json.dumps({"max_position_embeddings": 32}), encoding="utf-8"
    )
    (folder / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    record = {
        "name": identity.MODEL_NAME,
        "version": identity.MODEL_VERSION,
        "base_model": identity.BASE_MODEL,
        "revision": identity.PINNED_REVISION,
        "license": identity.LICENSE,
        "parameters": 6,
        "context_limit": 32,
        "weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "weight_bytes": weights.stat().st_size,
        "dtypes": ["F32"],
    }
    (folder / identity.IDENTITY_FILE).write_text(json.dumps(record), encoding="utf-8")
    identity._verified.cache_clear()
    identity.load_identity(verify_weights=True)
    return folder
