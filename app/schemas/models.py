"""The model the chat answers with, and the words it expects next."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HuggingFaceTrainBody(BaseModel):
    """A base model to fine-tune, and how many steps to fine-tune it for."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    steps: int = Field(default=200, ge=1, le=2000)


class PredictBody(BaseModel):
    """The text to read, and the model that should read it."""

    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(max_length=4000)
    model: str | None = None
    limit: int = Field(default=5, ge=1, le=10)


class TransferBody(BaseModel):
    """The teacher whose knowledge is transferred, and how much to ask it."""

    model_config = ConfigDict(protected_namespaces=())

    folder: str
    limit: int = Field(default=40, ge=1, le=200)
