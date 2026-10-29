"""Which licensed dataset each skill is measured from, and who shapes its records.

The registry is the only place a source's licence, dataset name and row budget
are written down, so the builder and the bank manifest cannot disagree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.services.assessment import (
    intake,
    intake_grammar,
    intake_reading,
    intake_reasoning,
    intake_writing,
)

SPLIT_SALT = "luke-assessment-v1"
PAGE_ROWS = 100

# The revisions of the three files read directly, pinned so a rebuild reads the
# same bytes the first build read.
REVISION_IFEVAL = "966cd89545d6b6acfd7638bc708b98261ca58e84"
REVISION_CRUXEVAL = "b96af0450242eb4da433032b90998f25588a5d0f"
REVISION_DOLLY = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"

# The BLiMP paradigms chosen for their range: agreement and case first, then
# movement, negation and island constraints, which is where the skill gets hard.
BLIMP_PARADIGMS = (
    "anaphor_number_agreement",
    "determiner_noun_agreement_1",
    "regular_plural_subject_verb_agreement_1",
    "irregular_past_participle_verbs",
    "transitive",
    "causative",
    "passive_1",
    "existential_there_quantifiers_2",
    "matrix_question_npi_licensor_present",
    "sentential_negation_npi_licensor_present",
    "wh_questions_object_gap",
    "complex_NP_island",
)

Shaper = Callable[[Mapping[str, Any], int], intake.Draft | None]


@dataclass(frozen=True)
class Fetch:
    """How many rows to read from one split, and whether to spread them out.

    A source that ships a line-delimited file is read straight from it at a
    pinned revision instead of through the paginated row server, which keeps a
    rebuild to one request per file.
    """

    config: str
    split: str
    rows: int
    spread: bool = True
    file: str = ""
    revision: str = ""


@dataclass(frozen=True)
class Source:
    """One licensed dataset, the skill it measures, and its shaper."""

    meta: intake.SourceMeta
    fetches: tuple[Fetch, ...]
    shaper: Shaper


def _meta(key: str, skill: str, dataset: str, licence: str, records: str) -> intake.SourceMeta:
    return intake.SourceMeta(
        key=key,
        skill=skill,
        dataset=dataset,
        licence=licence,
        url=f"https://huggingface.co/datasets/{dataset}",
        records=records,
    )


def _source(meta: intake.SourceMeta, shaper: Shaper, fetches: Iterable[Fetch]) -> Source:
    return Source(meta, tuple(fetches), shaper)


def _one_fetch(
    config: str,
    split: str,
    rows: int,
    file: str = "",
    revision: str = "",
) -> tuple[Fetch, ...]:
    return (Fetch(config=config, split=split, rows=rows, file=file, revision=revision),)


BLIMP = _source(
    _meta("blimp", "grammar", "nyu-mll/blimp", "CC BY 4.0", "paradigm and pair id"),
    intake_grammar.blimp_draft,
    (Fetch(config=name, split="train", rows=30, spread=False) for name in BLIMP_PARADIGMS),
)
IFEVAL = _source(
    _meta("ifeval", "conversation", "google/IFEval", "Apache-2.0", "the dataset's own key"),
    intake_grammar.ifeval_draft,
    _one_fetch("default", "train", 200, "ifeval_input_data.jsonl", REVISION_IFEVAL),
)
SQUAD = _source(
    _meta("squad", "reading", "rajpurkar/squad", "CC BY-SA 4.0", "the dataset's own id"),
    intake_reading.squad_draft,
    _one_fetch("plain_text", "validation", 300),
)
BOOLQ = _source(
    _meta("boolq", "reading", "google/boolq", "CC BY-SA 3.0", "row index"),
    intake_reading.boolq_draft,
    _one_fetch("default", "validation", 300),
)
DOLLY_SUMMARY = _source(
    _meta(
        "dolly_summary", "reading", "databricks/databricks-dolly-15k", "CC BY-SA 3.0", "row index"
    ),
    intake_writing.dolly_summary_draft,
    _one_fetch("default", "train", 1500, "databricks-dolly-15k.jsonl", REVISION_DOLLY),
)
GSM8K = _source(
    _meta("gsm8k", "mathematics", "openai/gsm8k", "MIT", "row index"),
    intake_reasoning.gsm8k_draft,
    _one_fetch("main", "test", 300),
)
CRUXEVAL = _source(
    _meta("cruxeval", "programming", "cruxeval-org/cruxeval", "MIT", "the dataset's own id"),
    intake_reasoning.cruxeval_draft,
    _one_fetch("default", "test", 400, "test.jsonl", REVISION_CRUXEVAL),
)
ARC_EASY = _source(
    _meta("arc_easy", "general_knowledge", "allenai/ai2_arc", "CC BY-SA 4.0", "the item's own id"),
    intake_reasoning.arc_easy_draft,
    _one_fetch("ARC-Easy", "test", 300),
)
ARC_CHALLENGE = _source(
    _meta("arc_challenge", "science", "allenai/ai2_arc", "CC BY-SA 4.0", "the item's own id"),
    intake_reasoning.arc_challenge_draft,
    _one_fetch("ARC-Challenge", "test", 300),
)
DOLLY_CREATIVE = _source(
    _meta(
        "dolly_creative",
        "creative_writing",
        "databricks/databricks-dolly-15k",
        "CC BY-SA 3.0",
        "row index",
    ),
    intake_writing.dolly_creative_draft,
    _one_fetch("default", "train", 1500, "databricks-dolly-15k.jsonl", REVISION_DOLLY),
)

SOURCES: dict[str, Source] = {
    source.meta.key: source
    for source in (
        BLIMP,
        IFEVAL,
        SQUAD,
        BOOLQ,
        DOLLY_SUMMARY,
        GSM8K,
        CRUXEVAL,
        ARC_EASY,
        ARC_CHALLENGE,
        DOLLY_CREATIVE,
    )
}


def shape_pool(source: Source, records: Iterable[tuple[int, Mapping[str, Any]]]) -> list[dict]:
    """Every record that shapes cleanly, each with its own provenance.

    A record whose draft names a different skill than the source is dropped
    rather than trusted: the registry, not the shaper, decides what a skill is.
    """
    items: list[dict] = []
    seen: set[str] = set()
    for index, record in records:
        draft = source.shaper(record, index)
        if draft is None or draft.skill != source.meta.skill or draft.source != source.meta.key:
            continue
        item = intake.build_item(draft, source.meta)
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        items.append(item)
    return items


def skill_sources() -> dict[str, tuple[str, ...]]:
    """Which source keys feed which skill, in registry order."""
    grouped: dict[str, list[str]] = {}
    for source in SOURCES.values():
        grouped.setdefault(source.meta.skill, []).append(source.meta.key)
    return {skill: tuple(keys) for skill, keys in grouped.items()}
