"""The small trained heads that read a message.

The heads sit on top of the embedding of a text: one reads the feeling, one
decides whether to ask a question back, and one judges whether an answer is
likely to be right. They are trained from the tenant's own material, so the
labels are the tenant's words.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.services import brain_paths, embeddings
from app.services.brain_text import tokenize
from app.services.heads import LinearHead

if TYPE_CHECKING:
    from app.services.language.training_materials import TrainingSet

LOGGER = logging.getLogger(__name__)

_EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "happy": (
        "happy",
        "glad",
        "pleased",
        "joy",
        "joyful",
        "excited",
        "cheerful",
        "smile",
        "delight",
    ),
    "sad": ("sad", "unhappy", "upset", "cry", "crying", "lonely", "sorry", "hurt"),
    "angry": ("angry", "mad", "annoyed", "annoying", "cross", "furious", "upset"),
    "tired": ("tired", "sleepy", "exhausted", "weary", "sleep"),
    "scared": ("scared", "afraid", "frightened", "nervous", "worried", "fear"),
    "surprised": ("surprised", "shock", "shocked", "amazed", "wow", "astonished"),
    "calm": ("calm", "quiet", "peaceful", "relaxed", "fine", "good", "gentle"),
}


def emotion_of(text: str) -> str:
    """The feeling a piece of text carries, using the taught emotion words."""
    words = set(tokenize(text))
    for emotion, markers in _EMOTION_WORDS.items():
        if words & set(markers):
            return emotion
    return "neutral"


def emotion_training_sentences(
    sentences: list[str], lessons: list[dict] | None = None
) -> tuple[list[str], list[str]]:
    """Label the taught sentences and lessons by the feeling they express."""
    labelled: list[str] = []
    labels: list[str] = []
    for sentence in sentences:
        emotion = emotion_of(sentence)
        if emotion == "neutral" and len(sentence.split()) > 3:
            continue
        labelled.append(sentence)
        labels.append(emotion)
    for lesson in lessons or []:
        for text in (lesson.get("question", ""), lesson.get("answer", "")):
            if not text:
                continue
            emotion = emotion_of(text)
            if emotion == "neutral":
                continue
            labelled.append(text)
            labels.append(emotion)
    return labelled, labels


def embedding_features(texts: list[str]) -> list[list[float]]:
    """The embedding of every text, in the space the collections are stored in."""
    if not texts:
        return []
    try:
        return embeddings.embed_texts(texts)
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.warning("The texts could not be embedded: %s", error)
        return []


def train_emotion_head(tenant_id: int | None, sentences: list[str]) -> int:
    """Teach the emotion head from the tenant's own sentences."""
    labels = ["neutral", *sorted(_EMOTION_WORDS)]
    head = LinearHead(labels)
    texts, emotions = emotion_training_sentences(sentences, None)
    features = embedding_features(texts)
    targets = [labels.index(emotion) for emotion in emotions]
    if features and len(features) == len(targets) and head.fit(features, targets):
        head.save(brain_paths.brain_path(tenant_id, "emotion"))
    return len(texts)


def train_feedback_heads(tenant_id: int | None, feedback: list[dict[str, Any]]) -> None:
    """Teach the heads that read a question and judge the answer."""
    clarify = LinearHead(["answer", "ask_back"])
    correct = LinearHead(["correct", "wrong"])
    questions = [entry["question"] for entry in feedback if entry.get("question")]
    features = embedding_features(questions)
    if not features or len(features) != len(feedback):
        return
    labels = [0 if entry.get("correct") else 1 for entry in feedback]
    clarify.fit(features, labels)
    correct.fit(features, labels)
    clarify.save(brain_paths.brain_path(tenant_id, "clarify"))
    correct.save(brain_paths.brain_path(tenant_id, "correct"))


def train_intent_head(tenant_id: int | None, materials: TrainingSet) -> int:
    """Teach the intent head from the tenant's own lessons."""
    from app.services.reasoning import intent as intent_service

    cards = [material.as_card() for material in materials.materials if material.question.strip()]
    return intent_service.train(tenant_id, cards)


def feel(tenant_id: int | None, text: str) -> dict[str, Any]:
    """The emotion in a message, from the trained head (or the taught words)."""
    head = LinearHead.load(brain_paths.brain_path(tenant_id, "emotion"))
    if head is not None:
        features = embedding_features([text])
        if features:
            label, confidence, scores = head.predict(features)
            if confidence >= 0.45:
                return {
                    "emotion": label,
                    "confidence": confidence,
                    "scores": scores,
                    "source": "trained",
                }
    return {"emotion": emotion_of(text), "confidence": 0.4, "scores": {}, "source": "words"}


def should_ask_back(tenant_id: int | None, question: str) -> dict[str, Any]:
    """Whether the assistant should ask a question back instead of answering."""
    head = LinearHead.load(brain_paths.brain_path(tenant_id, "clarify"))
    if head is not None:
        features = embedding_features([question])
        if features:
            label, confidence, _scores = head.predict(features)
            if label == "ask_back" and confidence >= 0.6:
                return {"ask_back": True, "confidence": confidence, "source": "trained"}
            return {"ask_back": False, "confidence": confidence, "source": "trained"}
    return {"ask_back": False, "confidence": 0.0, "source": "untrained"}


def answer_confidence(tenant_id: int | None, question: str) -> dict[str, Any]:
    """How likely the answer is to be right, from the trainer's corrections."""
    head = LinearHead.load(brain_paths.brain_path(tenant_id, "correct"))
    if head is not None:
        features = embedding_features([question])
        if features:
            label, confidence, _scores = head.predict(features)
            return {
                "likely_correct": label == "correct",
                "confidence": confidence,
                "source": "trained",
            }
    return {"likely_correct": True, "confidence": 0.5, "source": "untrained"}
