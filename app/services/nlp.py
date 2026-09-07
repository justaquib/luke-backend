import spacy
import re
from functools import lru_cache
from app.config import settings


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(settings.spacy_model)
    except OSError:
        return spacy.blank("en")


def preprocess(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    text = preprocess(text)
    nlp = get_nlp()

    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + sent).strip()
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current)
    return chunks


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    if "ner" not in nlp.pipe_names:
        return []
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]