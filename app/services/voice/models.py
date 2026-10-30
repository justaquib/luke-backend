"""The swappable voice-module protocol.

A voice module bundles a TTS engine and an STT engine behind one interface.
The registry holds every module Luke knows about; the active module is the one
the tenant has chosen in settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.services.voice.errors import VoiceModuleUnavailableError


@dataclass(frozen=True)
class VoiceInfo:
    """One voice a TTS engine can speak in."""

    id: str
    label: str
    language: str
    gender: str


@dataclass(frozen=True)
class VoiceModuleEntry:
    """A registered voice module: its name and the engines it bundles."""

    name: str
    tts_module: str
    stt_module: str
    description: str


class TtsEngine(Protocol):
    """What a TTS engine must do: list voices and turn text into WAV bytes."""

    def list_voices(self) -> list[VoiceInfo]: ...

    def synthesize(self, text: str, voice: str | None) -> bytes: ...


class SttEngine(Protocol):
    """What an STT engine must do: turn audio bytes into text."""

    def transcribe(self, audio_bytes: bytes, mime: str) -> str: ...


@dataclass(frozen=True)
class VoiceModule:
    """A paired TTS + STT engine, ready to serve one tenant."""

    entry: VoiceModuleEntry
    tts: TtsEngine
    stt: SttEngine


_REGISTRY: dict[str, tuple[type[TtsEngine], type[SttEngine]]] = {}
_ACTIVE: str | None = None
_STATE: dict[str, str | None] = {"active": None}


@dataclass(frozen=True)
class ModuleRegistration:
    """Parameters for registering a voice module."""

    name: str
    tts_module: str
    stt_module: str
    description: str
    tts_cls: type[TtsEngine]
    stt_cls: type[SttEngine]


def register_module(registration: ModuleRegistration) -> None:
    """Add a module to the registry. Called once at import time per module."""
    _REGISTRY[registration.name] = (registration.tts_cls, registration.stt_cls)
    _ensure_entry(
        registration.name,
        registration.tts_module,
        registration.stt_module,
        registration.description,
    )


def list_modules() -> list[VoiceModuleEntry]:
    """Every registered module, in registration order."""
    return list(_ENTRIES.values())


def _ensure_entry(name: str, tts_module: str, stt_module: str, description: str) -> None:
    if name not in _ENTRIES:
        _ENTRIES[name] = VoiceModuleEntry(
            name=name,
            tts_module=tts_module,
            stt_module=stt_module,
            description=description,
        )


_ENTRIES: dict[str, VoiceModuleEntry] = {}


def set_active_module(name: str) -> None:
    """Choose which registered module serves requests."""
    if name not in _REGISTRY:
        raise VoiceModuleUnavailableError(f"Unknown voice module: {name}")
    _STATE["active"] = name


def active_module() -> VoiceModule:
    """The module the tenant has chosen, built and ready to call."""
    name = _STATE["active"] or _default_name()
    if name is None:
        raise VoiceModuleUnavailableError("No voice module is registered.")
    if name not in _REGISTRY:
        raise VoiceModuleUnavailableError(f"Voice module not installed: {name}")
    tts_cls, stt_cls = _REGISTRY[name]
    entry = _ENTRIES[name]
    return VoiceModule(entry=entry, tts=tts_cls(), stt=stt_cls())


def _default_name() -> str | None:
    """The first registered module, when the tenant has not chosen one."""
    return next(iter(_REGISTRY), None)
