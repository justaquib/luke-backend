"""The whole ladder, from the seed to the generalist.

The ladder has 101 levels in eleven bands: ten steps in each of the ten bands
below 1.0, then the generalist milestone. Only the levels that have a test are
open; the rest are the road ahead, kept in the data so the page can show the
climb honestly, with every level beyond the tested frontier locked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.brain_levels import LUKE_LEVELS

FINAL_LEVEL = "1.0.0"

# The levels with a real test behind them. A level outside this set is road
# ahead: it has no probes, so it can never be measured or unlocked.
OPEN_LEVELS = frozenset(str(entry["level"]) for entry in LUKE_LEVELS)

_OPEN_BY_LEVEL = {str(entry["level"]): entry for entry in LUKE_LEVELS}


@dataclass(frozen=True)
class Band:
    """One tenth of the ladder, with its gate level and its nine steps."""

    key: str
    name: str
    gate: str
    detail: str
    rungs: tuple[str, ...]


_BANDS: tuple[Band, ...] = (
    Band(
        key="0.0",
        name="Foundations",
        gate="Seed",
        detail="Holding what it was taught, and answering from that material.",
        rungs=tuple(_OPEN_BY_LEVEL[f"0.0.{minor}"]["name"] for minor in range(1, 10)),
    ),
    Band(
        key="0.1",
        name="Independence",
        gate="Independent",
        detail="Answering a question it was never taught, in its own words.",
        rungs=(
            "Recalled",
            "Rephrased",
            "Recombined",
            "Rehearsed",
            "Drafted",
            "Revised",
            "Sourced",
            "Extended",
            "Self-taught",
        ),
    ),
    Band(
        key="0.2",
        name="Reasoning",
        gate="Reasoner",
        detail="Working an answer out instead of remembering it.",
        rungs=(
            "Compared",
            "Classified",
            "Sequenced",
            "Inferred",
            "Justified",
            "Weighed",
            "Hypothesised",
            "Debugged",
            "Argued",
        ),
    ),
    Band(
        key="0.3",
        name="Synthesis",
        gate="Synthesiser",
        detail="Bringing several taught things into one answer.",
        rungs=(
            "Summarised",
            "Combined",
            "Contrasted",
            "Mapped",
            "Structured",
            "Illustrated",
            "Recounted",
            "Condensed",
            "Unified",
        ),
    ),
    Band(
        key="0.4",
        name="Application",
        gate="Practitioner",
        detail="Using what it knows to do a job, not only to reply.",
        rungs=(
            "Applied",
            "Adapted",
            "Simulated",
            "Planned",
            "Executed",
            "Measured",
            "Troubleshot",
            "Optimised",
            "Delivered",
        ),
    ),
    Band(
        key="0.5",
        name="Adaptation",
        gate="Adapter",
        detail="Fitting what it knows to a situation it has not seen.",
        rungs=(
            "Transferred",
            "Reframed",
            "Personalised",
            "Simplified",
            "Expanded",
            "Localised",
            "Translated",
            "Restyled",
            "Recovered",
        ),
    ),
    Band(
        key="0.6",
        name="Composition",
        gate="Composer",
        detail="Writing at length, in its own voice, with a plan.",
        rungs=(
            "Outlined",
            "Composed",
            "Narrated",
            "Styled",
            "Edited",
            "Finished",
            "Published",
            "Serialised",
            "Curated",
        ),
    ),
    Band(
        key="0.7",
        name="Evaluation",
        gate="Evaluator",
        detail="Checking quality and finding its own mistakes.",
        rungs=(
            "Checked",
            "Verified",
            "Critiqued",
            "Ranked",
            "Audited",
            "Stressed",
            "Certified",
            "Challenged",
            "Appraised",
        ),
    ),
    Band(
        key="0.8",
        name="Mastery",
        gate="Master",
        detail="Holding a subject whole and teaching it to someone else.",
        rungs=(
            "Retained",
            "Reused",
            "Taught",
            "Mentored",
            "Coordinated",
            "Led",
            "Delegated",
            "Moderated",
            "Stewarded",
        ),
    ),
    Band(
        key="0.9",
        name="Judgment",
        gate="Judge",
        detail="Choosing well when the answer is not in the material.",
        rungs=(
            "Balanced",
            "Principled",
            "Contextual",
            "Ethical",
            "Accountable",
            "Decided",
            "Advised",
            "Reconciled",
            "Trusted",
        ),
    ),
)

_FINAL_BAND = Band(
    key="1.0",
    name="Generalist",
    gate="Generalist",
    detail="Sound judgment across everything it has been taught.",
    rungs=(),
)


def _band_fields(band: Band) -> dict[str, Any]:
    return {"band": band.key, "band_name": band.name, "band_detail": band.detail}


def _locked_description(band: Band, name: str, gate: bool) -> str:
    if gate:
        return (
            f"The gate into the {band.name} band. {band.detail} "
            "No test has been written for it yet, so it stays locked."
        )
    return f"{name}: {band.detail} No test has been written for this level yet."


def _entry(band: Band, index: int) -> dict[str, Any]:
    level = f"{band.key}.{index}"
    name = band.gate if index == 0 else band.rungs[index - 1]
    open_entry = _OPEN_BY_LEVEL.get(level)
    if open_entry is not None:
        return {**open_entry, "open": True, **_band_fields(band)}
    return {
        "level": level,
        "name": name,
        "description": _locked_description(band, name, index == 0),
        "reference": "Planned level: the ladder has not opened it yet.",
        "open": False,
        **_band_fields(band),
    }


def _build_roadmap() -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for band in _BANDS:
        levels.extend(_entry(band, index) for index in range(10))
    levels.append(
        {
            "level": FINAL_LEVEL,
            "name": _FINAL_BAND.gate,
            "description": (
                f"The top of the ladder. {_FINAL_BAND.detail} "
                "No test has been written for it yet, so it stays locked."
            ),
            "reference": "Planned milestone: the ladder has not opened it yet.",
            "open": False,
            **_band_fields(_FINAL_BAND),
        }
    )
    return levels


ROADMAP: list[dict[str, Any]] = _build_roadmap()
