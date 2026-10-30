"""HTTP routes for the voice module: status, TTS, STT, and module switching.

The router stays thin: it checks the caller, reads the request, and hands the
work to the voice service. Errors from the service are translated into HTTP
responses the frontend can show.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User
from app.schemas.voice import (
    SetModuleRequest,
    SttResponse,
    TtsRequest,
    VoiceInfoOut,
    VoiceModuleEntryOut,
    VoiceStatusOut,
)
from app.services import voice_settings
from app.services.voice import (
    VoiceModuleUnavailableError,
    VoiceSynthesisFailedError,
    VoiceTranscriptionFailedError,
    active_module,
    list_modules,
    set_active_module,
)

router = APIRouter(prefix="/voice", tags=["voice"])


@router.get("/status", response_model=VoiceStatusOut)
def voice_status(
    user: User = Depends(get_bearer_user),
) -> VoiceStatusOut:
    """What the settings page shows: the active module, its voices, and any error."""
    tenant_id = tenant_id_of(user)
    active_name = voice_settings.active_module_name(tenant_id)
    modules = list_modules()
    module_entry = next((m for m in modules if m.name == active_name), None)
    try:
        voice_settings.ensure_module_for(tenant_id)
        module = active_module()
        voices = [VoiceInfoOut.model_validate(v) for v in module.tts.list_voices()]
        return VoiceStatusOut(
            available=True,
            active_module=module_entry.name if module_entry else active_name,
            tts_module=module_entry.tts_module if module_entry else None,
            stt_module=module_entry.stt_module if module_entry else None,
            voices=voices,
            modules=[VoiceModuleEntryOut.model_validate(m) for m in modules],
            error=None,
        )
    except VoiceModuleUnavailableError as error:
        return VoiceStatusOut(
            available=False,
            active_module=active_name,
            tts_module=module_entry.tts_module if module_entry else None,
            stt_module=module_entry.stt_module if module_entry else None,
            voices=[],
            modules=[VoiceModuleEntryOut.model_validate(m) for m in modules],
            error=str(error),
        )


@router.post("/tts")
def voice_tts(
    body: TtsRequest,
    user: User = Depends(get_bearer_user),
) -> Response:
    """Synthesize text into audio and stream it back as a WAV file."""
    tenant_id = tenant_id_of(user)
    voice_settings.ensure_module_for(tenant_id)
    try:
        module = active_module()
        wav_bytes = module.tts.synthesize(body.text, body.voice)
    except VoiceModuleUnavailableError as error:
        raise HTTPException(503, str(error)) from error
    except VoiceSynthesisFailedError as error:
        raise HTTPException(400, str(error)) from error
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=luke-tts.wav"},
    )


@router.post("/stt", response_model=SttResponse)
async def voice_stt(
    audio: UploadFile,
    user: User = Depends(get_bearer_user),
) -> SttResponse:
    """Transcribe uploaded audio and return the text."""
    tenant_id = tenant_id_of(user)
    voice_settings.ensure_module_for(tenant_id)
    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/wav"
    try:
        module = active_module()
        text = module.stt.transcribe(audio_bytes, mime)
    except VoiceModuleUnavailableError as error:
        raise HTTPException(503, str(error)) from error
    except VoiceTranscriptionFailedError as error:
        raise HTTPException(400, str(error)) from error
    return SttResponse(text=text)


@router.get("/modules", response_model=list[VoiceModuleEntryOut])
def voice_modules(
    user: User = Depends(get_bearer_user),
) -> list[VoiceModuleEntryOut]:
    """Every registered voice module the tenant can switch to."""
    return [VoiceModuleEntryOut.model_validate(m) for m in list_modules()]


@router.put("/module", response_model=VoiceStatusOut)
def voice_set_module(
    body: SetModuleRequest,
    user: User = Depends(get_bearer_user),
) -> VoiceStatusOut:
    """Switch the active voice module for this tenant."""
    tenant_id = tenant_id_of(user)
    try:
        set_active_module(body.name)
        voice_settings.set_active_module_name(tenant_id, body.name)
    except VoiceModuleUnavailableError as error:
        raise HTTPException(400, str(error)) from error
    return voice_status(user)
