"""Tests for the grade tests over the language model routes.

A mock test is scored and never advances the grade, the final exam needs 80%,
and a grade test answers with the same lesson the preview answers with.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _start_and_score(
    client: TestClient, kind: str, outcomes: list[bool], stage: str = "nursery"
) -> dict:
    started = client.post("/language-model/test/start", json={"stage": stage, "kind": kind})
    assert started.status_code == 200, started.text
    assert started.json()["state"]["active_test"]["stage"] == stage
    assert started.json()["state"]["active_test"]["kind"] == kind
    for passed in outcomes:
        asked = client.post("/language-model/test/question", json={"question": "What is a drum?"})
        assert asked.status_code == 200, asked.text
        scored = client.post("/language-model/test/score", json={"passed": passed})
        assert scored.status_code == 200, scored.text
    finished = client.post("/language-model/test/finish", json={"passed": True})
    assert finished.status_code == 200, finished.text
    return finished.json()


def test_a_mock_test_is_scored_and_never_advances_the_grade(language_client: TestClient):
    finished = _start_and_score(language_client, "mock", [True])

    result = finished["result"]
    assert result["kind"] == "mock"
    assert result["passed"] is False, "a mock test never advances the grade"
    assert result["required_pass_rate"] is None
    assert result["answers_passed"] == 1
    assert finished["state"]["current_stage_id"] == "nursery"


def test_the_final_exam_needs_eighty_percent(language_client: TestClient):
    failed = _start_and_score(language_client, "exam", [True, True, True, False])

    assert failed["result"]["kind"] == "exam"
    assert failed["result"]["pass_rate"] == 0.75
    assert failed["result"]["required_pass_rate"] == 0.8
    assert failed["result"]["passed"] is False
    assert failed["state"]["current_stage_id"] == "nursery"

    passed = _start_and_score(language_client, "exam", [True, True, True, True])

    assert passed["result"]["pass_rate"] == 1.0
    assert passed["result"]["passed"] is True
    assert passed["state"]["current_stage_id"] == "lkg"


def test_a_passed_exam_cannot_be_taken_again(language_client: TestClient):
    _start_and_score(language_client, "exam", [True])

    response = language_client.post(
        "/language-model/test/start", json={"stage": "nursery", "kind": "exam"}
    )

    assert response.status_code == 400
    assert "already been passed" in response.text


def test_the_grade_test_answers_with_the_same_lesson_as_the_preview(language_client: TestClient):
    language_client.post(
        "/language-model/lessons",
        json={
            "lesson": "Q: What is a drum?\nA: A drum is something you hit with a stick.",
            "stage": "nursery",
            "subject": "Music",
        },
    )
    preview = language_client.post("/language-model/preview", json={"question": "What is a drum?"})
    assert preview.status_code == 200
    preview_answer = preview.json()
    assert preview_answer["source"] == "lesson_cards"

    started = language_client.post(
        "/language-model/test/start", json={"stage": "nursery", "kind": "mock"}
    )
    assert started.status_code == 200
    asked = language_client.post(
        "/language-model/test/question", json={"question": "What is a drum?"}
    )
    assert asked.status_code == 200
    test_answer = asked.json()

    assert test_answer["source"] == preview_answer["source"]
    assert test_answer["answer"] == preview_answer["answer"]
    assert "hit with a stick" in test_answer["answer"]

    language_client.post("/language-model/test/score", json={"passed": True})
    finished = language_client.post("/language-model/test/finish", json={"passed": True})
    assert finished.status_code == 200
    assert finished.json()["state"]["active_test"] is None
