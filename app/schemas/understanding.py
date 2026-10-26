"""What the backend understood and connected before it answered."""

from pydantic import BaseModel, Field


class WordReadingOut(BaseModel):
    """One word of the question, and what the offline dictionary says of it."""

    word: str
    lemma: str = ""
    part_of_speech: str = ""
    meaning: str = ""
    synonyms: list[str] = Field(default_factory=list)


class ConnectionOut(BaseModel):
    """One source consulted, with the measured reason for its outcome."""

    source: str
    outcome: str
    detail: str
    matched_question: str = ""


class QuestionUnderstandingOut(BaseModel):
    """The reading, the plan, the connections and the checks of one answer."""

    words: list[WordReadingOut] = Field(default_factory=list)
    question_type: str = ""
    key_words: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    connections: list[ConnectionOut] = Field(default_factory=list)
    grammar_notes: list[str] = Field(default_factory=list)
    verified: bool = False
    verification: str = ""
