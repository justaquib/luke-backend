"""Voice module: text-to-speech and speech-to-text for Luke.

The voice module is swappable. The active module is chosen per-tenant in the
settings file, so a tenant can run MeloTTS + Whisper today and switch to a
different HuggingFace stack later without touching the routes.
"""

from app.services.voice.errors import (
    VoiceModuleUnavailableError,
    VoiceSynthesisFailedError,
    VoiceTranscriptionFailedError,
)
from app.services.voice.models import (
    ModuleRegistration,
    VoiceInfo,
    VoiceModule,
    VoiceModuleEntry,
    active_module,
    list_modules,
    register_module,
    set_active_module,
)
from app.services.voice.stt import WhisperSttEngine
from app.services.voice.tts import MeloTtsEngine

register_module(
    ModuleRegistration(
        name="melo-whisper",
        tts_module="facebook/mms-tts-eng",
        stt_module="openai/whisper-tiny",
        description="Minimal English TTS (MeloTTS/MMS) + English STT (Whisper tiny).",
        tts_cls=MeloTtsEngine,
        stt_cls=WhisperSttEngine,
    )
)

__all__ = [
    "ModuleRegistration",
    "VoiceInfo",
    "VoiceModule",
    "VoiceModuleEntry",
    "VoiceModuleUnavailableError",
    "VoiceSynthesisFailedError",
    "VoiceTranscriptionFailedError",
    "active_module",
    "list_modules",
    "register_module",
    "set_active_module",
]
