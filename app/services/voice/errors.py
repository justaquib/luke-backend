"""Errors the voice module can raise.

Each error maps to one HTTP status in the router, so the frontend can show a
useful message instead of a generic 500.
"""

from __future__ import annotations


class VoiceModuleUnavailableError(RuntimeError):
    """The chosen voice module is not installed or failed to load.

    The router returns 503 with this message so the settings page can offer an
    install prompt instead of a broken voice card.
    """


class VoiceSynthesisFailedError(RuntimeError):
    """TTS could not produce audio for the given text."""


class VoiceTranscriptionFailedError(RuntimeError):
    """STT could not transcribe the given audio."""
