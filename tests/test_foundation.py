"""Luke's foundation: the identity it reports, the prompt it reads, the gate.

No test loads the real checkpoint. The identity tests build a small stand-in
with a real safetensors header, so the parameter count, the size and the hash
are all measured from files, exactly as they are in production.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import QueryRequest, QueryResponse
from app.services import answer_flow
from app.services.foundation import answers as foundation_answers
from app.services.foundation import identity, prompt, verify


def _build_checkpoint(root: Path, parameters: int = 6, context: int = 32768) -> Path:
    """A stand-in checkpoint with a readable header and a recorded identity."""
    folder = root / "Qwen2.5-0.5B-Instruct" / identity.PINNED_REVISION
    folder.mkdir(parents=True)
    header = json.dumps({"w": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 24]}})
    payload = header.encode()
    weights = len(payload).to_bytes(8, "little") + payload + b"\0" * 24
    (folder / "model.safetensors").write_bytes(weights)
    (folder / "config.json").write_text(
        json.dumps({"max_position_embeddings": context}), encoding="utf-8"
    )
    record = {
        "name": identity.MODEL_NAME,
        "version": identity.MODEL_VERSION,
        "base_model": identity.BASE_MODEL,
        "revision": identity.PINNED_REVISION,
        "license": identity.LICENSE,
        "parameters": parameters,
        "context_limit": context,
        "weight_sha256": identity.sha256_file(folder / "model.safetensors"),
        "weight_bytes": len(weights),
        "dtypes": ["BF16"],
    }
    (folder / "identity.json").write_text(json.dumps(record), encoding="utf-8")
    return folder


def test_the_identity_is_measured_from_the_files(tmp_path) -> None:
    folder = _build_checkpoint(tmp_path)

    found = identity.load_identity(folder)

    assert found.parameters == 6
    assert found.context_limit == 32768
    assert found.weight_dtype == "bf16"
    assert found.display == "Luke 0.0.1 · Qwen2.5-0.5B-Instruct · revision 7ae5576"


def test_a_missing_checkpoint_names_the_install_command(tmp_path) -> None:
    with pytest.raises(identity.FoundationMissingError) as error:
        identity.load_identity(tmp_path / "nothing")

    assert "install_foundation.py" in str(error.value)


def test_a_tampered_weight_file_is_refused(tmp_path) -> None:
    folder = _build_checkpoint(tmp_path)
    weights = folder / "model.safetensors"
    content = bytearray(weights.read_bytes())
    content[-1] = 1
    weights.write_bytes(bytes(content))

    with pytest.raises(identity.FoundationCorruptError):
        identity.load_identity(folder)


def test_a_recorded_parameter_count_that_does_not_match_is_refused(tmp_path) -> None:
    folder = _build_checkpoint(tmp_path, parameters=7)

    with pytest.raises(identity.FoundationCorruptError):
        identity.load_identity(folder)


def test_a_different_base_model_is_refused(tmp_path) -> None:
    folder = _build_checkpoint(tmp_path)
    record = json.loads((folder / "identity.json").read_text(encoding="utf-8"))
    record["base_model"] = "someone/else"
    (folder / "identity.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(identity.FoundationCorruptError):
        identity.load_identity(folder)


def test_the_status_reports_a_missing_checkpoint_without_raising(tmp_path) -> None:
    status = identity.status()

    assert status["installed"] is False
    assert status["identity"] is None
    assert "install_foundation.py" in status["error"]


def test_the_system_prompt_says_who_luke_is_and_what_it_may_claim(tmp_path) -> None:
    found = identity.load_identity(_build_checkpoint(tmp_path))

    system = prompt.system_prompt(found, "Nova", prompt.Persona(), ['Opening: "hi" → "Hello."'], 0)

    assert "You are Nova" in system
    assert "Qwen2.5-0.5B-Instruct" in system
    assert "never claim" in system
    assert "Do not invent a citation" in system
    assert 'Opening: "hi" → "Hello."' in system


def test_the_system_prompt_requires_citations_when_evidence_is_given(tmp_path) -> None:
    found = identity.load_identity(_build_checkpoint(tmp_path))

    system = prompt.system_prompt(found, "Luke", prompt.Persona(), [], 2)

    assert "2 numbered pieces of evidence" in system
    assert "cite every claim" in system
    assert "not been taught it" in system


def test_the_messages_carry_the_recent_turns_and_the_numbered_evidence() -> None:
    history = [
        {"role": "user", "content": "What is rain?"},
        {"role": "assistant", "content": "Rain is water falling from clouds."},
        {"role": "tool", "content": "ignored"},
    ]
    evidence = [prompt.Evidence(1, "document", "weather.pdf", "Rain is liquid water.")]

    messages = prompt.build_messages("system", history, "  Where does it come from?  ", evidence)

    assert messages[0] == {"role": "system", "content": "system"}
    assert [item["role"] for item in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"].startswith("Evidence:\n[1] (document: weather.pdf)")
    assert messages[-1]["content"].endswith("Question: Where does it come from?")


def test_history_is_trimmed_to_the_budget_from_the_oldest_turn() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    trimmed = prompt.trim_history(messages, lambda turns: len(turns) * 10, 30)

    assert trimmed[0]["content"] == "system"
    assert [item["content"] for item in trimmed[1:]] == ["two", "three"]


def test_a_cut_answer_keeps_its_last_finished_sentence() -> None:
    assert prompt.trim_incomplete("Rain falls. It comes from clouds and") == "Rain falls."
    assert prompt.trim_incomplete("Rain falls.") == "Rain falls."
    assert prompt.clean_answer("Assistant: Hello.") == "Hello."


def test_the_verifier_accepts_a_cited_answer_and_refuses_the_rest() -> None:
    evidence = ["Rain is liquid water that falls from clouds. It forms at 0 degrees."]

    assert verify.verify_grounded("Rain falls from clouds [1].", evidence)[0] is True
    assert verify.verify_grounded("Rain falls from clouds.", evidence)[0] is False
    assert verify.verify_grounded("Rain falls from clouds [2].", evidence)[0] is False
    introduced = verify.verify_grounded("Rain falls 400 metres from clouds [1].", evidence)
    assert introduced[0] is False
    assert "400" in introduced[1]


def _pending(db, req, entities, disabled, tenant_id=None):
    return QueryResponse(
        question=req.question,
        source="pending",
        answer="not learned",
        hits=[],
        entities=entities,
    )


def _quiet_sources(monkeypatch):
    from app.services import answer_sources

    monkeypatch.setattr(answer_sources.vector_store, "search_faq", lambda *args, **kwargs: [])
    monkeypatch.setattr(answer_sources.vector_store, "search_faq_loose", lambda *args, **kwargs: [])
    monkeypatch.setattr(answer_sources.vector_store, "query", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        answer_sources.answering, "answer_from_curriculum", lambda question, tenant: None
    )


def _written_stream(req, tenant_id, entities, model_name, cancel=None):
    yield "writing", {"model": model_name, "mode": "generated"}
    yield "token", {"text": "Rain is "}
    yield (
        "answer",
        QueryResponse(
            question=req.question,
            source="generated",
            answer="Rain is water falling from clouds.",
            hits=[],
            entities=entities,
            mode="generated",
        ),
    )


def test_a_question_without_taught_material_is_written_and_labelled(monkeypatch) -> None:
    _quiet_sources(monkeypatch)
    monkeypatch.setattr(foundation_answers, "installed", lambda: True)
    monkeypatch.setattr(foundation_answers, "stream", _written_stream)

    events = list(
        answer_flow.iter_answer_events(QueryRequest(question="What is rain?"), None, None, _pending)
    )

    answer = next(data for event, data in events if event == "answer")
    assert answer["source"] == "generated"
    assert answer["mode"] == "generated"
    assert answer["answer"] == "Rain is water falling from clouds."
    assert answer["understanding"]["verified"] is False
    assert "base checkpoint" in answer["understanding"]["verification"]
    assert any(event == "token" for event, _ in events)


def test_a_taught_answer_is_never_replaced_by_a_written_one(monkeypatch) -> None:
    from app.services import answer_sources
    from app.services.connections import Connection
    from app.services.language import answering

    _quiet_sources(monkeypatch)
    monkeypatch.setattr(foundation_answers, "installed", lambda: True)
    monkeypatch.setattr(foundation_answers, "stream", _written_stream)
    taught = answering.TaughtAnswer(
        answer="The waterfall is in the park.",
        source="lesson_cards",
        matched_question="Where is the waterfall?",
        evidence=(Connection("lesson_cards", "match", "The lesson covers it."),),
    )
    monkeypatch.setattr(
        answer_sources.answering, "answer_from_curriculum", lambda question, tenant: taught
    )

    events = list(
        answer_flow.iter_answer_events(
            QueryRequest(question="Where is the waterfall?"), None, None, _pending
        )
    )

    answer = next(data for event, data in events if event == "answer")
    assert answer["source"] == "lesson_cards"
    assert answer["mode"] == "retrieved"
    assert not any(event == "token" for event, _ in events)


def test_a_failed_foundation_falls_back_to_the_pending_answer(monkeypatch) -> None:
    from app.services.foundation.identity import FoundationError

    _quiet_sources(monkeypatch)
    monkeypatch.setattr(foundation_answers, "installed", lambda: True)

    def broken(*args: object, **kwargs: object):
        raise FoundationError("The checkpoint cannot be read.")
        yield

    monkeypatch.setattr(foundation_answers, "stream", broken)

    events = list(
        answer_flow.iter_answer_events(QueryRequest(question="What is rain?"), None, None, _pending)
    )

    answer = next(data for event, data in events if event == "answer")
    assert answer["source"] == "pending"
    details = [item.get("detail", "") for event, item in events if event == "connecting"]
    assert any("cannot be read" in detail for detail in details)
