"""Pydantic schemas for the voice module endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceInfoOut(BaseModel):
    """One voice a TTS engine can speak in."""

    id: str
    label: str
    language: str
    gender: str


class VoiceModuleEntryOut(BaseModel):
    """A registered voice module: its name and the engines it bundles."""

    name: str
    tts_module: str
    stt_module: str
    description: str


class VoiceStatusOut(BaseModel):
    """What the settings page shows about the active voice module."""

    available: bool
    active_module: str | None
    tts_module: str | None
    stt_module: str | None
    voices: list[VoiceInfoOut]
    modules: list[VoiceModuleEntryOut]
    error: str | None = None


class TtsRequest(BaseModel):
    """Text to synthesize and the voice to speak it in."""

    text: str = Field(..., min_length=1, max_length=2000)
    voice: str | None = None


class SttResponse(BaseModel):
    """The transcription of an uploaded audio clip."""

    text: str


class SetModuleRequest(BaseModel):
    """Which voice module the tenant wants to use."""

    name: str = Field(..., min_length=1)
