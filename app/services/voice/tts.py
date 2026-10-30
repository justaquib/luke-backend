"""Text-to-speech engine backed by a HuggingFace model.

The default is ``facebook/mms-tts-eng``, a minimal English-only checkpoint from
the Massively Multilingual Speech project. It is small (~60MB), fast on CPU,
and needs no system-level dependencies beyond torch and transformers, which
Luke already ships.

The engine is lazy: the model is downloaded and loaded on the first call to
``synthesize``, so the backend starts even when the checkpoint is not yet on
disk. A failed load raises ``VoiceModuleUnavailableError`` with a message the
settings page can show.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from app.services.voice.errors import VoiceModuleUnavailableError, VoiceSynthesisFailedError
from app.services.voice.models import VoiceInfo

_DEFAULT_MODEL = "facebook/mms-tts-eng"
_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class _LoadedEngine:
    pipeline: object
    sample_rate: int


class MeloTtsEngine:
    """A TTS engine that calls a HuggingFace text-to-speech pipeline."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self._model = model
        self._loaded: _LoadedEngine | None = None
        self._load_error: str | None = None

    def list_voices(self) -> list[VoiceInfo]:
        """The speakers the checkpoint ships with.

        MMS-TTS-English has one speaker; multilingual checkpoints expose more
        and can override this method in a subclass.
        """
        return [
            VoiceInfo(
                id="default",
                label="English · Default",
                language="en",
                gender="female",
            )
        ]

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Turn text into a WAV buffer.

        The first call loads the model; subsequent calls reuse it. A load
        failure is remembered so the next call fails fast instead of retrying
        the download.
        """
        if not text.strip():
            raise VoiceSynthesisFailedError("Cannot synthesize empty text.")
        engine = self._ensure_loaded()
        try:
            result = engine.pipeline(text)
        except Exception as error:
            self._load_error = None
            self._loaded = None
            raise VoiceSynthesisFailedError(f"TTS inference failed: {error}") from error
        audio_array = self._extract_audio(result)
        return self._to_wav(audio_array, engine.sample_rate)

    def _ensure_loaded(self) -> _LoadedEngine:
        if self._loaded is not None:
            return self._loaded
        if self._load_error is not None:
            raise VoiceModuleUnavailableError(self._load_error)
        try:
            from transformers import pipeline  # type: ignore[import-untyped]

            pipe = pipeline("text-to-speech", model=self._model)
            sample_rate = int(getattr(pipe, "sampling_rate", _SAMPLE_RATE))
        except Exception as error:
            self._load_error = f"Voice module {self._model} is not available: {error}"
            raise VoiceModuleUnavailableError(self._load_error) from error
        self._loaded = _LoadedEngine(pipeline=pipe, sample_rate=sample_rate)
        return self._loaded

    @staticmethod
    def _extract_audio(result: object) -> list[float]:
        """The raw samples from whatever shape the pipeline returned."""
        if isinstance(result, dict):
            for key in ("audio", "raw", "samples", "array"):
                if key in result:
                    value = result[key]
                    if isinstance(value, list):
                        return [float(sample) for sample in value]
                    try:
                        return [float(sample) for sample in value]  # type: ignore[arg-type]
                    except TypeError:
                        continue
        if hasattr(result, "audio"):
            return MeloTtsEngine._extract_audio({"audio": result.audio})
        raise VoiceSynthesisFailedError("TTS pipeline returned no audio.")

    @staticmethod
    def _to_wav(samples: list[float], sample_rate: int) -> bytes:
        """Encode float samples as a 16-bit PCM WAV buffer."""
        import torch

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            tensor = torch.tensor(samples, dtype=torch.float32).clamp(-1.0, 1.0)
            wav.writeframes((tensor * 32767).to(torch.int16).numpy().tobytes())
        return buffer.getvalue()
