"""A chunk id is stable, so re-indexing overwrites instead of duplicating."""

from __future__ import annotations

from app.services.chunk_vectors import chunk_id


def test_the_same_chunk_gets_the_same_id() -> None:
    first = chunk_id(0, "rain.txt", "Rain falls from clouds.")
    second = chunk_id(0, "rain.txt", "Rain falls from clouds.")

    assert first == second


def test_different_text_or_source_gets_a_different_id() -> None:
    assert chunk_id(0, "rain.txt", "Rain falls.") != chunk_id(0, "rain.txt", "Snow falls.")
    assert chunk_id(0, "rain.txt", "Rain falls.") != chunk_id(0, "snow.txt", "Rain falls.")


def test_two_workspaces_never_share_an_id() -> None:
    assert chunk_id(0, "rain.txt", "Rain falls.") != chunk_id(1, "rain.txt", "Rain falls.")


def test_an_unset_workspace_is_the_canonical_one() -> None:
    assert chunk_id(None, "rain.txt", "Rain falls.") == chunk_id(0, "rain.txt", "Rain falls.")
