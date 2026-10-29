"""Read-only compatibility checks on extracted packages, before any restore writes."""

from __future__ import annotations

import pickle
import re
import struct
import subprocess
import sys
from pathlib import Path

import torch

from app.services.foundation import identity
from app.services.language.package import compatibility_state
from app.services.language.package.checks import as_list, as_object, read_json, require
from app.services.language.package.manifest import PackageError, safe_member, sha256_path
from app.services.neural.model import WORD_MODEL_CONFIG, TinyTransformer
from app.services.neural.tokenizer import artifacts
from app.services.neural.tokenizer.base import (
    META_FILE,
    SPECIAL_NAMES,
    TOKENIZER_FILE,
    TokenizerError,
)
from app.services.neural.tokenizer.bpe import BpeTokenizer
from app.services.neural.tokenizer.word import WordTokenizer

_DIGEST = re.compile(r"[0-9a-f]{64}")
_READ_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    OverflowError,
    RuntimeError,
    AssertionError,
    struct.error,
)


def _paths(root: Path, payload: dict) -> set[str]:
    paths: set[str] = set()
    for record in as_list(payload.get("files"), dict, "Manifest files"):
        name = record.get("path")
        require(isinstance(name, str) and safe_member(name), "Invalid staged file path")
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()), f"Unsafe staged path {name}")
        require(path.is_file(), f"Missing staged file {name}")
        paths.add(name)
    return paths


def _native_tokenizer(path: Path) -> None:
    # tokenizers' Rust parser raises plain Exception, not a typed parse error.
    # Isolate that parser rather than broadly catching programming errors here.
    command = "from tokenizers import Tokenizer; import sys; Tokenizer.from_file(sys.argv[1])"
    try:
        subprocess.run(  # noqa: S603 -- fixed interpreter command, path passed as argv
            [sys.executable, "-c", command, str(path)],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise PackageError(
            f"Invalid tokenizer {path.name}; export its original artifact again."
        ) from error


def _record(record: dict, tokenizer: WordTokenizer | BpeTokenizer, digest: str) -> None:
    require(record.get("kind") == tokenizer.kind, "Tokenizer kind does not match its artifact")
    require(record.get("id") == digest, "Tokenizer id does not match its vocabulary or artifact")
    size = record.get("vocab_size")
    require(
        type(size) is int and size == tokenizer.vocab_size, "Tokenizer vocabulary size mismatch"
    )
    specials = as_object(record.get("special_ids"), "Tokenizer special_ids")
    require(set(specials) == set(SPECIAL_NAMES), "Tokenizer special marks are incomplete")
    for name, token_id in specials.items():
        require(
            type(token_id) is int and 0 <= token_id < size, f"Special id {name} is out of range"
        )
    require(specials == tokenizer.special_ids, "Tokenizer special ids disagree with its vocabulary")


def _artifacts(root: Path, payload: dict, paths: set[str]) -> dict[str, BpeTokenizer]:
    loaded: dict[str, BpeTokenizer] = {}
    for record in as_list(payload.get("tokenizers"), dict, "Manifest tokenizers"):
        artifact_id = record.get("id")
        require(
            isinstance(artifact_id, str) and bool(_DIGEST.fullmatch(artifact_id)),
            "Invalid tokenizer id",
        )
        require(artifact_id not in loaded, "Duplicate tokenizer artifact record")
        prefix = f"weights/tokenizers/{artifact_id}/"
        require(
            {prefix + TOKENIZER_FILE, prefix + META_FILE} <= paths,
            f"Missing BPE artifact {artifact_id}",
        )
        folder = root / prefix
        require(
            sha256_path(folder / TOKENIZER_FILE) == artifact_id,
            "Tokenizer artifact hash disagrees with folder id",
        )
        meta = as_object(read_json(folder / META_FILE), "Tokenizer metadata")
        require(
            meta.get("format") == artifacts.FORMAT and meta.get("schema") == artifacts.SCHEMA,
            "Unsupported tokenizer metadata schema",
        )
        data = as_object(read_json(folder / TOKENIZER_FILE), "Tokenizer JSON")
        require(
            as_object(data.get("model"), "Tokenizer model").get("type") == "BPE",
            "Scratch artifact is not BPE",
        )
        _native_tokenizer(folder / TOKENIZER_FILE)
        tokenizer = BpeTokenizer.load(folder)
        _record(meta, tokenizer, artifact_id)
        _record(
            {**record, "special_ids": record.get("special_ids", meta["special_ids"])},
            tokenizer,
            artifact_id,
        )
        require(
            meta.get("special_tokens") == list(tokenizer.special_ids),
            "Tokenizer metadata special tokens disagree",
        )
        loaded[artifact_id] = tokenizer
    folders = {Path(name).parts[2] for name in paths if name.startswith("weights/tokenizers/")}
    require(folders == set(loaded), "Staged tokenizer folders disagree with manifest records")
    return loaded


def _tensor(value: object, label: str) -> torch.Tensor:
    require(isinstance(value, torch.Tensor), f"{label} must be a tensor")
    require(
        value.layout == torch.strided and value.is_floating_point(),
        f"{label} must be a dense floating tensor",
    )
    require(
        value.device.type == "cpu" and bool(torch.isfinite(value).all()),
        f"{label} contains invalid weights",
    )
    return value


def _head(data: dict) -> None:
    labels = as_list(data.get("labels"), str, "Head labels")
    require(
        bool(labels) and len(set(labels)) == len(labels), "Head labels must be nonempty and unique"
    )
    weights = _tensor(data.get("weights"), "Head weights")
    bias = _tensor(data.get("bias"), "Head bias")
    require(
        weights.ndim == 2 and weights.shape[0] > 0 and weights.shape[1] == len(labels),
        "Head weights must have features by labels shape",
    )
    require(bias.shape == (len(labels),), "Head bias shape must match labels")
    require("state" not in data, "Head checkpoints use labels, weights and bias, not state")


def _model(data: dict, vocabulary: list[str]) -> dict:
    config = {**WORD_MODEL_CONFIG, **as_object(data.get("config"), "Checkpoint config")}
    for key in ("d_model", "heads", "layers", "context"):
        require(
            type(config[key]) is int and config[key] > 0,
            f"Checkpoint config {key} must be a positive integer",
        )
    require(
        config["d_model"] % config["heads"] == 0, "Checkpoint heads do not divide embedding width"
    )
    dropout = config["dropout"]
    require(type(dropout) in (int, float) and 0 <= dropout <= 1, "Invalid checkpoint dropout")
    require(
        config.get("vocabulary_size", len(vocabulary)) == len(vocabulary),
        "Config vocabulary size disagrees with checkpoint",
    )
    state = as_object(data.get("state"), "Checkpoint state")
    # A meta model checks every tensor shape without allocating untrusted config sizes.
    require(config["layers"] <= len(state), "Checkpoint config has more layers than weights")
    with torch.device("meta"):
        expected = TinyTransformer(len(vocabulary), config).state_dict()
    require(set(state) == set(expected), "Checkpoint state has missing or unexpected weights")
    for name, tensor in expected.items():
        found = _tensor(state[name], f"Checkpoint {name}")
        require(
            found.shape == tensor.shape, f"Checkpoint {name} shape disagrees with config/vocabulary"
        )
    return config


def _word(data: dict, loaded: dict[str, BpeTokenizer]) -> None:
    schema = data.get("schema", 1)
    require(type(schema) is int and schema in (1, 2, 3), "Unsupported word checkpoint schema")
    require(data.get("task", "words") == "words", "Checkpoint is not a word model")
    vocabulary = as_list(data.get("vocabulary"), str, "Checkpoint vocabulary")
    require(
        bool(vocabulary) and len(vocabulary) == len(set(vocabulary)),
        "Checkpoint vocabulary is empty or duplicated",
    )
    record = as_object(data.get("tokenizer", {}), "Checkpoint tokenizer")
    require(schema < 3 or bool(record), "Schema 3 checkpoint has no tokenizer record")
    kind = record.get("kind", "word")
    require(kind in ("word", "bpe"), "Unknown checkpoint tokenizer kind")
    if kind == "bpe":
        artifact_id = record.get("id")
        require(
            isinstance(artifact_id, str) and artifact_id in loaded,
            "Checkpoint references a missing staged BPE artifact",
        )
        tokenizer = loaded[artifact_id]
        digest = artifact_id
    else:
        tokenizer = WordTokenizer(vocabulary)
        digest = tokenizer.digest()
    if record:
        _record(record, tokenizer, digest)
    require(
        vocabulary == tokenizer.vocabulary(),
        "Checkpoint vocabulary differs from staged tokenizer vocabulary",
    )
    config = _model(data, vocabulary)
    require(
        config.get("tokenizer_kind", kind) == kind,
        "Config tokenizer kind disagrees with checkpoint",
    )
    for source in (data, config):
        for key, special in (
            ("pad_id", "<pad>"),
            ("start_id", "<start>"),
            ("end_id", "<end>"),
            ("unknown_id", "<unk>"),
        ):
            value = source.get(key, tokenizer.special_ids[special])
            require(
                type(value) is int and value == tokenizer.special_ids[special],
                f"Checkpoint {key} disagrees with tokenizer",
            )


def _checkpoints(root: Path, payload: dict, paths: set[str], loaded: dict) -> None:
    primary = payload.get("checkpoint")
    if primary is not None:
        primary = as_object(primary, "Manifest checkpoint")
        require(isinstance(primary.get("name"), str), "Manifest checkpoint has no name")
        require(
            f"weights/checkpoints/{primary['name']}" in paths,
            "Manifest checkpoint is missing from staging",
        )
    for name in sorted(paths):
        if not name.startswith("weights/checkpoints/"):
            continue
        data = as_object(torch.load(root / name, map_location="cpu", weights_only=True), name)
        if Path(name).name.startswith("words-"):
            _word(data, loaded)
        else:
            _head(data)
        if primary and primary["name"] == Path(name).name:
            for field in ("schema", "task", "tokenizer"):
                if field in primary:
                    require(
                        primary[field] == data.get(field, {} if field == "tokenizer" else None),
                        f"Manifest checkpoint {field} disagrees with staged checkpoint",
                    )


def _foundation(root: Path, payload: dict, paths: set[str]) -> None:
    base = as_object(payload.get("base_model"), "Manifest base_model")
    included = base.get("included")
    require(type(included) is bool, "Base inclusion must be a boolean")
    names = {name for name in paths if name.startswith("weights/base/")}
    require(bool(names) == included, "Foundation inclusion disagrees with staged files")
    require(payload.get("mode") in ("full", "slim"), "Unknown package mode")
    require(
        payload["mode"] != "slim" or not included, "Slim packages cannot carry foundation files"
    )
    require(
        base.get("base_model") == identity.BASE_MODEL
        and base.get("revision") == identity.PINNED_REVISION,
        "Package foundation reference is not the pinned model/revision",
    )
    if not included:
        return  # A slim reference is restorable even when no foundation is installed.
    required = {
        identity.WEIGHT_FILE,
        identity.IDENTITY_FILE,
        "config.json",
        TOKENIZER_FILE,
        "tokenizer_config.json",
    }
    require(
        {f"weights/base/{name}" for name in required} <= paths, "Foundation files are incomplete"
    )
    folder = root / "weights/base"
    for name in required - {identity.WEIGHT_FILE}:
        as_object(read_json(folder / name), f"Foundation {name}")
    _native_tokenizer(folder / TOKENIZER_FILE)
    found = identity.inspect_identity(folder)
    for key in ("base_model", "revision", "weight_sha256", "weight_bytes"):
        require(
            base.get(key) == found.as_dict()[key],
            f"Foundation {key} disagrees with staged identity",
        )


def validate(root: Path, payload: dict) -> None:
    """Validate staged weights, tokenizers and state for dry-run and actual import."""
    label = "package"
    try:
        as_object(payload, "Manifest")
        paths = _paths(root, payload)
        loaded = _artifacts(root, payload, paths)
        label = "checkpoint"
        _checkpoints(root, payload, paths, loaded)
        label = "foundation"
        _foundation(root, payload, paths)
        for name in sorted(paths):
            if name.startswith("state/"):
                label = name
                compatibility_state.state(root / name)
    except (
        *_READ_ERRORS,
        pickle.UnpicklingError,
        EOFError,
        TokenizerError,
        identity.FoundationError,
    ) as error:
        raise PackageError(
            f"Cannot validate {label}: {error}. Export intact compatible files again."
        ) from error
