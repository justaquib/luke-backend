"""Tests for measuring what the model writes, and for keeping the probe clean.

The old evaluation answered every probe by looking the answer up. These tests
pin the other behaviour: the writer answers the probe, and a held-out request
that crept into the training wording is reported rather than scored.
"""

from app.services.language import evaluation, instructions

POEM_CARD = {
    "id": "poem-1",
    "question": "Poem: Rain Rain Go Away",
    "answer": "Rain, rain, go away,\nCome again another day.\nLittle children want to play.",
    "subject": "Language, Poems",
    "subjects": ["Language", "Poems"],
    "grade_id": "lkg",
}

PROBE = instructions.Instruction(
    instruction="Write a short poem about the rain.",
    response=POEM_CARD["answer"],
    kind="poem",
    held_out=True,
)


class FakeWriter:
    """A writer that says what it is told and remembers being asked."""

    last_confidence = 0.9

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.asked: list[str] = []

    def respond(self, instruction, context="", max_words=64, reading=None):
        self.asked.append(instruction)
        return self.answer


def test_a_probe_is_answered_by_the_writer_when_there_is_one():
    writer = FakeWriter("Rain, rain, go away,\nCome again another day.")

    result = evaluation._probes(None, [POEM_CARD], [PROBE], writer=writer)

    probe = result[0]["probes"][0]
    assert writer.asked == [PROBE.instruction]
    assert probe["written"] is True
    assert probe["answered"] == writer.answer


def test_a_probe_without_a_writer_is_a_lookup():
    result = evaluation._probes(None, [POEM_CARD], [PROBE], writer=None)

    probe = result[0]["probes"][0]
    assert probe["written"] is False
    assert probe["answered"] == POEM_CARD["answer"]


def test_the_cross_form_score_counts_written_and_looked_up_separately():
    writer = FakeWriter(POEM_CARD["answer"])

    score = evaluation._cross_form(None, [POEM_CARD], writer, [PROBE])

    assert score["written"] == 1
    assert score["looked_up"] == 0
    assert score["total"] == 1
    assert score["score"] == 1.0


def test_cross_form_hands_the_writer_to_the_probes(monkeypatch):
    seen = {}

    def record(tenant_id, cards, items, writer=None):
        seen["writer"] = writer
        seen["items"] = items
        return [{"passed": 0, "total": 1, "probes": []}]

    monkeypatch.setattr(evaluation, "_probes", record)
    writer = object()

    evaluation._cross_form(None, [POEM_CARD], writer, [PROBE])

    assert seen["writer"] is writer
    assert seen["items"] == [PROBE]


def test_a_held_out_request_that_was_taught_is_reported_as_contamination():
    taught = instructions.training_instructions([POEM_CARD])

    report = evaluation._contamination([POEM_CARD], [taught[0]])

    assert report["checked"] == 1
    assert report["overlaps"] == 1
    assert report["clean"] is False
    assert report["examples"] == [taught[0].instruction]


def test_the_real_held_out_requests_are_never_part_of_the_training():
    held_out = instructions.held_out_requests([POEM_CARD])

    report = evaluation._contamination([POEM_CARD], held_out)

    assert report["checked"] == len(held_out)
    assert report["overlaps"] == 0
    assert report["clean"] is True


def test_an_untaught_request_is_refused_without_a_lesson():
    class Silent:
        last_confidence = 0.0

        def respond(self, instruction, context="", max_words=64, reading=None):
            return ""

    result = evaluation._refusals(
        None,
        [],
        Silent(),
        [("Write a poem about volcanoes.", evaluation.REFUSAL)],
    )

    assert result["passed"] == 1
    assert result["model"] == 1
    assert result["probes"][0]["correct"] is True
