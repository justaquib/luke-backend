"""The question a page wants the writer to read, and have measured."""

from pydantic import BaseModel, Field


class AnswerTraceBody(BaseModel):
    """The question whose reading by the writer should be traced."""

    question: str = Field(min_length=1, max_length=500)
