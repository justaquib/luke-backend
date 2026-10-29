"""A passage is scored, cleaned and fingerprinted before it enters the index."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.extraction import ExtractedText, PageText
from app.services.quality import pipeline
from app.services.quality.boilerplate import join_pages, without_boilerplate
from app.services.quality.density import text_density
from app.services.quality.score import passage_quality
from app.services.quality.scrub import scrub
from app.services.quality.simhash import hamming, simhash, to_signed, to_unsigned


def test_prose_is_dense_and_a_table_row_is_not():
    assert text_density("The rain falls on the green hills.") == 1.0
    assert text_density("| 1 | 2 | 3 |") == 0.0
    assert text_density("") == 0.0


def test_a_clean_english_sentence_scores_one():
    assert passage_quality("The rain falls on the green hills.") == 1.0


def test_a_sentence_that_repeats_itself_scores_zero():
    assert passage_quality("rain rain rain rain rain rain") == 0.0


def test_a_fragment_of_a_sentence_is_not_material():
    assert passage_quality("alpha beta") == 0.0


def test_a_passage_whose_language_is_unconfirmed_keeps_half_its_shape():
    assert passage_quality("Le chat est sur la table verte") == 0.5


def test_the_fingerprint_of_near_text_stays_close_and_far_text_moves():
    original = (
        "The rain falls on the green hills every morning, and the river runs to the sea "
        "through the valley where the old mill stands beside the road. Farmers bring their "
        "sheep down from the high fields before the winter, and the village prepares for "
        "the long cold months ahead with wood and grain."
    )
    near = original.replace("every morning", "every evening")
    far = (
        "A spreadsheet lists invoices, totals and the date each payment is due. The finance "
        "team checks every row against the bank statement before the quarter closes, and "
        "marks the accounts that are still open so the next review starts from the right place."
    )

    assert hamming(simhash(original), simhash(original)) == 0
    assert hamming(simhash(original), simhash(near)) <= 5
    assert hamming(simhash(original), simhash(far)) > 10
    assert simhash("") == 0


def test_a_fingerprint_survives_the_signed_range_of_a_bigint():
    top = (1 << 64) - 1

    assert to_signed(0) == 0
    assert to_signed(1 << 63) == -(1 << 63)
    assert to_unsigned(to_signed(top)) == top
    assert hamming(to_signed(top), 0) == 64


def _numbered_pages() -> ExtractedText:
    return join_pages(
        [PageText(1, "", 0, 0), PageText(2, "", 0, 0)],
        [
            "Page 1 of 2\nReal words here.\nAcme Corp",
            "Page 2 of 2\nMore real words.\nAcme Corp",
        ],
    )


def test_the_furniture_of_a_page_is_removed_and_the_spans_are_rebuilt():
    cleaned = without_boilerplate(_numbered_pages())

    assert cleaned.text == "Real words here.\nMore real words."
    assert [page.page for page in cleaned.pages] == [1, 2]
    assert cleaned.page_at(0) == 1
    assert cleaned.page_at(cleaned.text.index("More")) == 2
    for page in cleaned.pages:
        assert cleaned.text[page.char_start : page.char_end] == page.text


def test_a_bare_page_number_is_removed_only_when_it_is_the_page_number():
    pages = join_pages(
        [PageText(3, "", 0, 0)],
        ["3\nReal words on the third page."],
    )

    cleaned = without_boilerplate(pages)

    assert cleaned.text == "Real words on the third page."


def test_a_text_without_pages_keeps_a_line_that_looks_like_a_number():
    text = "42\nReal words after the number."
    extracted = ExtractedText(text=text, pages=(PageText(None, text, 0, len(text)),))

    assert without_boilerplate(extracted).text == text


def test_a_copyright_line_is_furniture_on_one_page():
    text = "© 2026 Acme Ltd\nReal words on the page."
    extracted = ExtractedText(text=text, pages=(PageText(1, text, 0, len(text)),))

    assert without_boilerplate(extracted).text == "Real words on the page."


def test_personal_details_are_masked():
    text = "Mail avery@example.com or call +1 (555) 123-4567. SSN 123-45-6789."

    masked = scrub(text)

    assert "avery@example.com" not in masked
    assert "[email]" in masked
    assert "[phone]" in masked
    assert "123-45-6789" not in masked
    assert "[id]" in masked


def test_the_scrub_is_off_until_the_trainer_turns_it_on(monkeypatch):
    text = "Mail avery@example.com."
    extracted = ExtractedText(text=text, pages=(PageText(None, text, 0, len(text)),))

    assert pipeline.prepare_document(extracted).text == text
    assert pipeline.prepare_text(text) == text

    monkeypatch.setattr(pipeline, "settings", SimpleNamespace(pii_scrub_enabled=True))
    assert "[email]" in pipeline.prepare_document(extracted).text
    assert "[email]" in pipeline.prepare_text(text)
