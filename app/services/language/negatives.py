"""The requests about things the learner was never taught, and the refusal.

The model has to learn where its material stops. A request about something the
trainer never taught is refused instead of answered with an unrelated lesson,
so the training shows the model a few wordings of such a request. The topics
below are concrete things a lesson could be about; a topic the material
already mentions is left out, so every one of these requests is genuinely
unknown to the tenant.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.services.language.topics import words_of

# How many requests about things that were never taught are given.
MAX_NEGATIVE_EXAMPLES = 200

# The answer the model gives when it was taught nothing about the request.
REFUSAL = "I do not know anything about that."

# The wordings an unknown request is made in, so the refusal is learned for
# more than one way of asking.
PHRASINGS = (
    "Write a poem about the {topic}.",
    "Tell me everything about the {topic}.",
    "Describe the {topic} to me.",
)

# Concrete things a lesson could be about. One that the material mentions is
# not used, so a known subject never teaches the model to refuse it.
UNKNOWN_TOPICS = (
    "accordion",
    "albatross",
    "alloy",
    "anemone",
    "archipelago",
    "aurora",
    "avalanche",
    "bagpipe",
    "bamboo",
    "basilisk",
    "beaver",
    "bison",
    "blizzard",
    "bobcat",
    "bonsai",
    "canyon",
    "capybara",
    "caribou",
    "cascade",
    "cheetah",
    "clarinet",
    "cobra",
    "comet",
    "coral",
    "crater",
    "crayfish",
    "cypress",
    "dolphin",
    "dragonfly",
    "dune",
    "eclipse",
    "egret",
    "estuary",
    "falcon",
    "fern",
    "fjord",
    "flamingo",
    "fossil",
    "gecko",
    "geyser",
    "glacier",
    "gondola",
    "granite",
    "heron",
    "hibiscus",
    "iceberg",
    "iguana",
    "jaguar",
    "juniper",
    "kayak",
    "kelp",
    "lagoon",
    "lemur",
    "lichen",
    "lighthouse",
    "llama",
    "lotus",
    "magnet",
    "mango",
    "marimba",
    "meadow",
    "meerkat",
    "meteor",
    "monsoon",
    "moose",
    "moss",
    "narwhal",
    "oasis",
    "obsidian",
    "orchid",
    "otter",
    "oyster",
    "panda",
    "panther",
    "papaya",
    "peacock",
    "pelican",
    "penguin",
    "plateau",
    "plum",
    "prairie",
    "puma",
    "quartz",
    "quokka",
    "rainforest",
    "raven",
    "reef",
    "saxophone",
    "sequoia",
    "sitar",
    "sloth",
    "sparrow",
    "spruce",
    "starfish",
    "summit",
    "swamp",
    "telescope",
    "thunder",
    "toucan",
    "tulip",
    "tundra",
    "turtle",
    "ukulele",
    "valley",
    "volcano",
    "walrus",
    "waterfall",
    "whale",
    "willow",
    "wombat",
    "zebra",
)


def _taught_words(cards: Sequence[dict[str, Any]]) -> set[str]:
    """Every word the lessons mention, in their questions, answers and subject."""
    return {
        word.lower()
        for card in cards
        for word in words_of(
            f"{card.get('question', '')} {card.get('answer', '')} {card.get('subject', '')}"
        )
    }


def unknown_topics(cards: Sequence[dict[str, Any]]) -> list[str]:
    """The things no lesson mentions, in the order they are listed."""
    taught = _taught_words(cards)
    return [topic for topic in UNKNOWN_TOPICS if topic not in taught]


def _all_negatives(cards: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    """Every unknown request, each topic in every wording."""
    return [
        (phrasing.format(topic=topic), REFUSAL)
        for topic in unknown_topics(cards)
        for phrasing in PHRASINGS
    ]


def build_negatives(cards: Sequence[dict[str, Any]]) -> list[tuple[str, str]]:
    """The unknown requests, each topic in every wording, up to the cap."""
    return _all_negatives(cards)[:MAX_NEGATIVE_EXAMPLES]


def build_held_out_negatives(cards: Sequence[dict[str, Any]], limit: int) -> list[tuple[str, str]]:
    """Unknown requests the training never showed, so refusal is measured honestly."""
    trained = set(build_negatives(cards))
    return [pair for pair in _all_negatives(cards) if pair not in trained][:limit]
