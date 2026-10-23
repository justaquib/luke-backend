"""The shapes a request can have.

A request is not a question: it asks for something to be written, and says
what about. The shapes are the wordings the model is trained on, so a request
is answered the way the model was taught to answer it.
"""

from __future__ import annotations

import re

# The kinds of writing a request can ask for.
ART_FORMS = ("poem", "rhyme", "song", "verse", "story", "letter", "sentence", "list")

# The shapes of a request. The topic is the part in the group, and the kind
# says which kind of lesson answers it.
REQUEST_SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:can|could|would|please)\s+you\s+(?:please\s+)?(?:write|compose|make|create|"
            r"recite|say|give)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:short\s+|little\s+|"
            r"small\s+|new\s+)?(?P<art>poem|rhyme|song|verse|story|letter|sentence|list)\s+"
            r"(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\b(?:write|compose|make(?:\s+up)?|create|recite|say|give)\s+(?:me\s+)?"
            r"(?:a\s+|an\s+|the\s+)?"
            r"(?:short\s+|little\s+|small\s+|new\s+)?"
            r"(?P<art>poem|rhyme|song|verse|story|letter|sentence|list)\s+"
            r"(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\bwrite\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:short\s+|little\s+|small\s+|new\s+)?"
            r"(?P<art>poem|rhyme|song|verse|story|letter|sentence|list)\s+"
            r"(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\bi would like (?:a|an|one)\s+(?P<art>poem|rhyme|song|verse|story|letter|"
            r"sentence|list)\s+(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\bplease write (?:a|an|one)\s+(?P<art>poem|rhyme|song|verse|story|letter|"
            r"sentence|list)\s+(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\bwrite something about\s+(?P<topic>[^.?]+)\s+in\s+a\s+"
            r"(?P<art>poem|rhyme|song|verse|story|letter|sentence|list)\b",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\bput\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?"
            r"(?P<art>poem|rhyme|song|verse|story|letter|sentence|list)\s+"
            r"(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+?)\s+in writing\b",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\bi need\s+(?:a\s+|an\s+|one\s+)?"
            r"(?P<art>poem|rhyme|song|verse|story|letter|sentence|list)\s+"
            r"(?:about|on|for)\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "art",
    ),
    (
        re.compile(
            r"\btell me\s+(?:everything\s+|all\s+|the details\s+|what you know\s+)?about\s+"
            r"(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bwhat can you say about\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bwhat do you know about\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bwhat can you tell me about\s+(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bgive me (?:information|some info|the details|a few facts|a bit)\s+about\s+"
            r"(?:the\s+)?(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bexplain\s+(?P<topic>[^.?]+)\s+to me\b",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bi (?:would like to know about|want to hear about|want to learn about)\s+"
            r"(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bwhat is there to know about\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\btell me about the\s+(?P<topic>[^.?]+)\s+please",
            re.IGNORECASE,
        ),
        "fact",
    ),
    (
        re.compile(
            r"\bgive me a list of\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\b(?:name|list)\s+(?:some|a few|me\s+some)\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\blist\s+(?:the\s+|me\s+the\s+|some\s+)?(?P<topic>[^.?]+?)\s+for me\b",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\b(?:give|tell) me\s+(?:some|a few)\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\bwhat are some\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\bplease list the\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\blist out\s+(?:some|a few)\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\bi need\s+(?:some|a few)\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\bi want a list of\s+(?P<topic>[^.?]+)",
            re.IGNORECASE,
        ),
        "list",
    ),
    (
        re.compile(
            r"\bgive me the\s+(?P<topic>[^.?]+)\s+please",
            re.IGNORECASE,
        ),
        "list",
    ),
)
