"""The guided coaching suggestions when the coach's answer misses.

The coach is learning to talk and to write. When its answer is not good
enough, the parent, or the trainer, is offered ways to coach it better: two
coaching phrasings to pick from, and the context the retry is made with.
"""

from __future__ import annotations

from dataclasses import dataclass

EMPATHY_OPENERS = (
    "That sounds really hard for your child.",
    "I hear how worried you are.",
    "It sounds like your child felt upset about this.",
)

ACTION_GUIDANCE = ("Stay calm and name the feeling before you set the limit.",)


@dataclass(frozen=True)
class CoachingScaffold:
    """What the parent can choose from when the coach's answer misses."""

    suggestions: list[str]
    retry_context: str
    failure: str


def build(answer: str, failure: list[str]) -> CoachingScaffold:
    """Two ways to say the coaching better, and why the answer was sent back."""
    suggestion_one = f"{EMPATHY_OPENERS[0]} You can say it calmly, in your own words."
    suggestion_two = f"{EMPATHY_OPENERS[2]} Name the feeling out loud before you set the limit."
    return CoachingScaffold(
        suggestions=[suggestion_one, suggestion_two],
        retry_context=f"The coach answered: {answer}.",
        failure=" ".join(failure) or "The coach was not sure of its answer.",
    )
