"""Speech-to-text engine backed by OpenAI Whisper.

The default is the ``tiny`` checkpoint: ~75MB on disk, fast enough for short
utterances on CPU, and good enough for English voice input to a desktop
assistant. Larger checkpoints (``base``, ``small``, ``medium``) can be chosen
in settings when the tenant trades speed for accuracy.

Like the TTS engine, the model is downloaded on first use and cached for the
life of the process.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass

from app.services.voice.errors import VoiceModuleUnavailableError, VoiceTranscriptionFailedError

_DEFAULT_MODEL = "tiny"


@dataclass(frozen=True)
class _LoadedEngine:
    model: object


class WhisperSttEngine:
    """An STT engine that calls a local Whisper checkpoint."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._loaded: _LoadedEngine | None = None
        self._load_error: str | None = None

    def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        """Turn an audio upload into text.

        Whisper's Python API wants a file path or a file-like object it can
        read with ``librosa``. The bytes are written to a temp file with the
        right suffix so the loader picks the correct decoder.
        """
        if not audio_bytes:
            raise VoiceTranscriptionFailedError("No audio was sent.")
        engine = self._ensure_loaded()
        suffix = _suffix_for(mime)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            try:
                result = engine.model.transcribe(tmp.name, language="en", fp16=False)
            except Exception as error:
                self._load_error = None
                self._loaded = None
                raise VoiceTranscriptionFailedError(
                    f"Whisper transcription failed: {error}"
                ) from error
        text = str(result.get("text", "")).strip()
        if not text:
            raise VoiceTranscriptionFailedError("Whisper returned no text.")
        return text

    def _ensure_loaded(self) -> _LoadedEngine:
        if self._loaded is not None:
            return self._loaded
        if self._load_error is not None:
            raise VoiceModuleUnavailableError(self._load_error)
        try:
            import whisper  # type: ignore[import-untyped]

            model = whisper.load_model(self._model, device="cpu")
        except Exception as error:
            self._load_error = f"Whisper model {self._model} is not available: {error}"
            raise VoiceModuleUnavailableError(self._load_error) from error
        self._loaded = _LoadedEngine(model=model)
        return self._loaded


def _suffix_for(mime: str) -> str:
    """The file extension Whisper needs for a given MIME type."""
    mime = mime.lower()
    if "wav" in mime:
        return ".wav"
    if "mpeg" in mime or "mp3" in mime:
        return ".mp3"
    if "ogg" in mime:
        return ".ogg"
    if "webm" in mime:
        return ".webm"
    if "m4a" in mime or "mp4" in mime:
        return ".m4a"
    return ".wav"


def _read_audio_bytes(source: bytes | io.BytesIO) -> bytes:
    """Accept either raw bytes or a file-like buffer."""
    if isinstance(source, bytes):
        return source
    return source.read()
