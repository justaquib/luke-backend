"""Loading and running Luke's foundation, one copy per process.

The checkpoint is read from its pinned folder and never modified. Generation is
serialised because one machine runs one model, and the stream stops as soon as
the client goes away instead of writing into a closed connection.
"""

from __future__ import annotations

import logging
import resource
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.foundation.identity import (
    FoundationError,
    FoundationIdentity,
    load_identity,
)
from app.services.foundation.prompt import fallback_template

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS_LIMIT = 512
STREAM_TIMEOUT = 180
_GENERATION_LOCK = threading.Lock()


@dataclass(frozen=True)
class GenerationParams:
    """How much and how freely the model writes."""

    max_new_tokens: int = 220
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    stop: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls) -> GenerationParams:
        """The configured defaults, with the token cap held to its limit."""
        return cls(
            max_new_tokens=min(settings.foundation_max_new_tokens, MAX_NEW_TOKENS_LIMIT),
            temperature=settings.foundation_temperature,
            top_p=settings.foundation_top_p,
            repetition_penalty=settings.foundation_repetition_penalty,
        )


@dataclass(frozen=True)
class GenerationResult:
    """One measured generation."""

    text: str
    prompt_tokens: int
    new_tokens: int
    seconds: float
    stop_reason: str

    @property
    def tokens_per_second(self) -> float:
        """The measured writing speed of this run."""
        return round(self.new_tokens / self.seconds, 3) if self.seconds else 0.0


class FoundationModel:
    """The loaded checkpoint, its tokenizer and the device it runs on."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        identity: FoundationIdentity,
        device: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.identity = identity
        self.device = device

    def chat_text(self, messages: Sequence[dict[str, str]]) -> str:
        """The checkpoint's own chat template, or the documented fallback."""
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, ValueError, TypeError):
            return fallback_template(messages)

    def count_tokens(self, messages: Sequence[dict[str, str]]) -> int:
        """The prompt size the model will actually read."""
        encoded = self.tokenizer(self.chat_text(messages), add_special_tokens=False)
        return len(encoded["input_ids"])

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        params: GenerationParams | None = None,
        cancel: threading.Event | None = None,
    ) -> GenerationResult:
        """Write one answer and return it with its measured speed."""
        import torch

        settings_for_run = params or GenerationParams.from_settings()
        encoded = self._encode(messages, settings_for_run.max_new_tokens)
        start = time.perf_counter()
        with _GENERATION_LOCK, torch.inference_mode():
            output = self.model.generate(**self._kwargs(encoded, settings_for_run, cancel))
        seconds = time.perf_counter() - start
        prompt_tokens = int(encoded["input_ids"].shape[1])
        new_tokens = int(output.shape[1]) - prompt_tokens
        written = self.tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
        return GenerationResult(
            text=_trim_stop(written, settings_for_run.stop),
            prompt_tokens=prompt_tokens,
            new_tokens=new_tokens,
            seconds=round(seconds, 3),
            stop_reason=_stop_reason(output[0][prompt_tokens:], settings_for_run, cancel),
        )

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        params: GenerationParams | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield the answer as it is written, and stop when the caller stops."""
        from transformers import TextIteratorStreamer

        settings_for_run = params or GenerationParams.from_settings()
        event = cancel or threading.Event()
        encoded = self._encode(messages, settings_for_run.max_new_tokens)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=STREAM_TIMEOUT,
        )
        errors: list[BaseException] = []

        def run() -> None:
            import torch

            try:
                with _GENERATION_LOCK, torch.inference_mode():
                    self.model.generate(
                        **self._kwargs(encoded, settings_for_run, event),
                        streamer=streamer,
                    )
            except BaseException as error:  # noqa: BLE001 -- the generation worker boundary
                # Process boundary: a worker that dies must end the stream, or the
                # reader waits on a generator that will never finish.
                logger.warning("the foundation worker stopped while writing: %s", error)
                errors.append(error)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True, name="luke-foundation")
        worker.start()
        try:
            for chunk in streamer:
                if event.is_set():
                    break
                if chunk:
                    yield chunk
        finally:
            event.set()
            worker.join(timeout=10)
        if errors and not isinstance(errors[0], (KeyboardInterrupt, SystemExit)):
            raise FoundationError(f"Luke's foundation stopped while writing: {errors[0]}")

    def _encode(self, messages: Sequence[dict[str, str]], max_new_tokens: int) -> Any:
        """The tokenised prompt, trimmed to leave room for the answer."""
        text = self.chat_text(messages)
        limit = max(64, self.identity.context_limit - max_new_tokens)
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=limit,
        ).to(self.device)

    def _kwargs(
        self,
        encoded: Any,
        params: GenerationParams,
        cancel: threading.Event | None,
    ) -> dict[str, Any]:
        """The generation arguments, with a stop check when a cancel exists."""
        kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.temperature > 0,
            "repetition_penalty": params.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = max(params.temperature, 0.01)
            kwargs["top_p"] = params.top_p
        if cancel is not None:
            from transformers import StoppingCriteriaList

            kwargs["stopping_criteria"] = StoppingCriteriaList([lambda *_: cancel.is_set()])
        return kwargs


class _ModelCache:
    """The one loaded model of this process, guarded by one lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model: FoundationModel | None = None
        self.adapter_folder: str | None = None
        self.adapter_label: str | None = None

    def get(self, adapter: str | Path | None = None) -> FoundationModel:
        """Load on first use, and reload only when a different adapter is asked for.

        A plain ``load()`` returns whatever this process already serves: the
        adapter is dropped only when a caller names a different one, never as a
        side effect of asking for the model again.
        """
        wanted = None if adapter is None else str(Path(adapter).expanduser().resolve())
        with self.lock:
            if self.model is not None and (wanted is None or self.adapter_folder == wanted):
                return self.model
            self.model = None
            self.adapter_folder = None
            self.adapter_label = None
            label = adapter_label(wanted) if wanted else None
            self.model = _load_model(wanted)
            self.adapter_folder = wanted
            self.adapter_label = label
            return self.model

    def clear(self) -> None:
        """Forget the model without unloading torch's own copy."""
        with self.lock:
            self.model = None
            self.adapter_folder = None
            self.adapter_label = None


_CACHE = _ModelCache()


def load(adapter: str | Path | None = None) -> FoundationModel:
    """The one loaded model of this process, loaded on first use.

    An adapter is attached only when a caller names its folder; the pinned
    checkpoint on disk is never modified, and a process that asks for no
    adapter serves exactly what it served before.
    """
    return _CACHE.get(adapter)


def adapter() -> str | None:
    """The label of the adapter the loaded model carries, or none."""
    return _CACHE.adapter_label


def loaded() -> bool:
    """Whether the model is already in memory, without loading it."""
    return _CACHE.model is not None


def loaded_device() -> str | None:
    """The device the loaded model runs on, or nothing when it is not loaded."""
    model = _CACHE.model
    return model.device if model is not None else None


def identity() -> FoundationIdentity:
    """The verified identity of the installed checkpoint."""
    return load_identity()


def unload() -> None:
    """Drop the model from this process, for tests and measurement scripts."""
    _CACHE.clear()


def memory_bytes() -> int:
    """This process's peak resident memory, in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_model(adapter: str | Path | None = None) -> FoundationModel:
    """Read the verified checkpoint and put it on the best available device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identity_for_load = load_identity()
    tokenizer = AutoTokenizer.from_pretrained(identity_for_load.folder)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(identity_for_load.folder, dtype=dtype)
        model.to(device)
    except RuntimeError as error:
        logger.warning("The %s load failed (%s); falling back to the CPU.", device, error)
        device = "cpu"
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(identity_for_load.folder, dtype=dtype)
        model.to(device)
    model.eval()
    if adapter is not None:
        model = _attach_adapter(model, Path(adapter), identity_for_load, dtype, device)
    return FoundationModel(model, tokenizer, identity_for_load, device)


def adapter_label(folder: str | Path) -> str:
    """Which adapter this folder holds, named by its weights' own hash."""
    from app.services.foundation.identity import sha256_file

    path = Path(folder)
    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FoundationError(
            f"{path} is not an adapter folder: it holds no adapter_config.json "
            "and adapter_model.safetensors."
        )
    return f"{path.name}#{sha256_file(weights)[:16]}"


def _attach_adapter(
    model: Any,
    folder: Path,
    identity_for_load: FoundationIdentity,
    dtype: Any,
    device: str,
) -> Any:
    """Put a saved adapter on the checkpoint it was trained against, or refuse it."""
    import json

    from peft import PeftModel

    record = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    trained_on = str(record.get("base_model_name_or_path") or "")
    if trained_on != identity_for_load.base_model:
        raise FoundationError(
            f"The adapter in {folder} was trained on {trained_on or 'an unnamed base'}, "
            f"not on {identity_for_load.base_model}; it is not loaded."
        )
    wrapped = PeftModel.from_pretrained(model, str(folder))
    wrapped.to(device=device, dtype=dtype)
    wrapped.eval()
    return wrapped


def _trim_stop(text: str, stops: tuple[str, ...]) -> str:
    """The answer up to the first stop string the caller asked for."""
    for stop in stops:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def _stop_reason(
    new_ids: Any,
    params: GenerationParams,
    cancel: threading.Event | None,
) -> str:
    """Why writing stopped: the caller, the token cap or the end token."""
    if cancel is not None and cancel.is_set():
        return "cancelled"
    if int(new_ids.shape[0]) >= params.max_new_tokens:
        return "max_new_tokens"
    return "eos"
