"""Serving a saved adapter over the pinned checkpoint, and naming it in a score.

No test loads the real checkpoint or a real adapter. The folder checks and the
cache are exercised directly; the attachment itself is refused before `peft`
sees a model when the adapter was trained on another base.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.assessment import conditions
from app.services.foundation import runtime


def _adapter_folder(root: Path, base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Path:
    folder = root / "candidate-qwen"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model, "r": 8}), encoding="utf-8"
    )
    (folder / "adapter_model.safetensors").write_bytes(b"adapter weights")
    return folder


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    runtime._CACHE.clear()
    yield
    runtime._CACHE.clear()


def test_an_adapter_is_named_by_its_folder_and_its_weights_hash(tmp_path: Path) -> None:
    folder = _adapter_folder(tmp_path)
    label = runtime.adapter_label(folder)
    assert label.startswith("candidate-qwen#")
    assert label.rsplit("#", 1)[-1] == runtime.adapter_label(folder).rsplit("#", 1)[-1]
    assert len(label.rsplit("#", 1)[-1]) == 16


def test_a_folder_that_is_not_an_adapter_is_refused_by_name(tmp_path: Path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(runtime.FoundationError, match="not an adapter folder"):
        runtime.adapter_label(folder)


def test_an_adapter_trained_on_another_base_is_refused(tmp_path: Path) -> None:
    folder = _adapter_folder(tmp_path, base_model="some/other-model")
    identity = SimpleNamespace(base_model="Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(runtime.FoundationError, match="trained on some/other-model"):
        runtime._attach_adapter(object(), folder, identity, None, "cpu")


def test_the_cache_keeps_one_model_and_reloads_for_a_different_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _adapter_folder(tmp_path / "first", "Qwen/Qwen2.5-0.5B-Instruct")
    second = _adapter_folder(tmp_path / "second", "Qwen/Qwen2.5-0.5B-Instruct")
    loads: list[str | None] = []

    def fake_load(adapter: str | None = None) -> SimpleNamespace:
        loads.append(adapter)
        return SimpleNamespace(adapter=adapter)

    monkeypatch.setattr(runtime, "_load_model", fake_load)
    one = runtime.load(first)
    again = runtime.load(first)
    two = runtime.load(second)
    assert one is again
    assert two is not one
    assert loads == [str(first.resolve()), str(second.resolve())]
    assert runtime.adapter() == runtime.adapter_label(second)


def test_a_plain_load_keeps_the_adapter_already_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _adapter_folder(tmp_path)
    loads: list[str | None] = []

    def fake_load(adapter: str | None = None) -> SimpleNamespace:
        loads.append(adapter)
        return SimpleNamespace(adapter=adapter)

    monkeypatch.setattr(runtime, "_load_model", fake_load)
    attached = runtime.load(folder)
    assert runtime.load() is attached
    assert loads == [str(folder.resolve())]
    assert runtime.adapter() == runtime.adapter_label(folder)


def test_a_run_without_an_adapter_names_only_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = SimpleNamespace(
        name="Luke",
        version="0.0.1",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        revision="abc123",
        weight_sha256="0123456789abcdef",
        folder=tmp_path,
    )
    monkeypatch.setattr("app.services.foundation.identity.load_identity", lambda *args: identity)
    monkeypatch.setattr(runtime, "adapter", lambda: None)
    artifact = conditions.provenance()["model_artifact"]
    assert artifact == "Qwen/Qwen2.5-0.5B-Instruct@abc123#0123456789abcdef"


def test_a_run_with_an_adapter_names_the_adapter_in_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = SimpleNamespace(
        name="Luke",
        version="0.0.1",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        revision="abc123",
        weight_sha256="0123456789abcdef",
        folder=tmp_path,
    )
    monkeypatch.setattr("app.services.foundation.identity.load_identity", lambda *args: identity)
    monkeypatch.setattr(runtime, "adapter", lambda: "candidate-qwen#fedcba9876543210")
    artifact = conditions.provenance()["model_artifact"]
    assert artifact == (
        "Qwen/Qwen2.5-0.5B-Instruct@abc123#0123456789abcdef+adapter:candidate-qwen#fedcba9876543210"
    )
