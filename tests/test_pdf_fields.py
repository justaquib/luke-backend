import json
import unittest
from unittest.mock import Mock, patch

import pymupdf

from app.core.pdf_fields import (
    _extract_pdf_field_values,
    answer_from_document,
    faq_field_subjects_compatible,
)
from app.services import chunk_vectors, vector_store


class PdfFieldAnswerTests(unittest.TestCase):
    def make_voucher_page(self, customer_name: str = "Avery Sample-"):
        document = pymupdf.open()
        page = document.new_page(width=600, height=400)
        # Form headings share a row while their values are aligned beneath them.
        for x, label in (
            (40, "CUSTOMER NAME"),
            (168, "AGENT"),
            (298, "MOBILE NO."),
            (426, "EMAIL"),
        ):
            page.insert_text((x, 120), label, fontsize=9)
        page.insert_text((40, 139), customer_name, fontsize=10)
        page.insert_text((40, 151), "Person", fontsize=10)
        page.insert_text((168, 139), "Jordan Helper", fontsize=10)
        page.insert_text((298, 139), "5551234567", fontsize=10)
        page.insert_text((426, 139), "avery@example.test", fontsize=10)
        page.insert_text((40, 220), "RESERVATION CODE", fontsize=9)
        page.insert_text((300, 220), "STATUS", fontsize=9)
        page.insert_text((40, 239), "AB-2048", fontsize=10)
        page.insert_text((300, 239), "Confirmed", fontsize=10)
        return document, page

    def make_hit(self, fields: dict[str, str], text: str = "Voucher details"):
        return {
            "text": text,
            "metadata": {
                "source": "voucher.pdf",
                "page_number": 1,
                "pdf_fields": json.dumps(fields),
            },
            "distance": 0.2,
        }

    def test_extracts_wrapped_customer_name_and_other_table_fields(self):
        document, page = self.make_voucher_page()
        try:
            fields = _extract_pdf_field_values(
                page,
                page_text="Voucher No: 002\nBooking Date: 17 Sep 2026\nPolicy Code: ZX-14",
            )
        finally:
            document.close()

        self.assertEqual(fields["customer_name"], "Avery SamplePerson")
        self.assertEqual(fields["agent"], "Jordan Helper")
        self.assertEqual(fields["mobile_no"], "5551234567")
        self.assertEqual(fields["email"], "avery@example.test")
        self.assertEqual(fields["reservation_code"], "AB-2048")
        self.assertEqual(fields["status"], "Confirmed")
        self.assertEqual(fields["voucher_no"], "002")
        self.assertEqual(fields["booking_date"], "17 Sep 2026")
        self.assertEqual(fields["policy_code"], "ZX-14")

    def test_question_context_does_not_become_part_of_field_name(self):
        hit = self.make_hit({"customer_name": "Avery SamplePerson"})
        answer = answer_from_document(
            "What is the customer name in the confirmation voucher?", [hit]
        )
        self.assertEqual(answer, "Customer name: Avery SamplePerson")

    def test_agent_question_prefers_agent_and_never_falls_back_to_customer_name(self):
        question = "What is the name of the agent in the confirmation voucher?"
        with_agent = self.make_hit(
            {"customer_name": "Avery SamplePerson", "agent": "Jordan Helper"}
        )
        without_agent = self.make_hit({"customer_name": "Avery SamplePerson"})

        self.assertEqual(answer_from_document(question, [with_agent]), "Agent: Jordan Helper")
        self.assertEqual(
            answer_from_document(question, [without_agent]),
            "I couldn't reliably identify the agent in this document. Please check "
            "the source passage.",
        )

    def test_faq_matching_rejects_a_different_requested_field(self):
        self.assertFalse(
            faq_field_subjects_compatible(
                "What is the name of the agent in the confirmation voucher?",
                "What is the customer name in the confirmation voucher?",
            )
        )
        self.assertTrue(
            faq_field_subjects_compatible(
                "What is the customer name in the voucher?",
                "Who is the customer named on this confirmation voucher?",
            )
        )

    def test_voucher_number_and_booking_date_are_returned_as_labeled_values(self):
        hit = self.make_hit({"voucher_no": "002", "booking_date": "17 Sep 2026"})
        self.assertEqual(
            answer_from_document("What is the voucher number?", [hit]),
            "Voucher no: 002",
        )
        self.assertEqual(
            answer_from_document("What was the booking date in the voucher?", [hit]),
            "Booking date: 17 Sep 2026",
        )

    def test_other_positioned_field_labels_can_be_asked_in_context(self):
        hit = self.make_hit({"reservation_code": "AB-2048"})
        self.assertEqual(
            answer_from_document("What is the reservation code on this voucher?", [hit]),
            "Reservation code: AB-2048",
        )

    @patch("app.services.chunk_vectors.get_collection")
    def test_pdf_fields_are_saved_as_chunk_metadata(self, get_collection):
        collection = Mock()
        get_collection.return_value = collection
        chunk_vectors.add_chunks(
            "voucher.pdf",
            ["Voucher text"],
            ["voucher-1"],
            tenant_id=7,
            notes=chunk_vectors.ChunkNotes(
                page_numbers=[1],
                pdf_fields=[{"customer_name": "Avery SamplePerson"}],
            ),
        )

        metadata = collection.upsert.call_args.kwargs["metadatas"][0]
        self.assertEqual(metadata["source"], "voucher.pdf")
        self.assertEqual(metadata["tenant_id"], "7")
        self.assertEqual(metadata["page_number"], 1)
        self.assertEqual(
            json.loads(metadata["pdf_fields"]), {"customer_name": "Avery SamplePerson"}
        )

    @patch("app.services.vector_store.get_collection")
    def test_chunk_search_fuses_semantic_and_word_matches(self, get_collection):
        documents = [
            "Ferry schedule and island transfer details.",
            "Confirmation voucher. CUSTOMER NAME: Avery SamplePerson",
        ]
        metadatas = [
            {"source": "travel.pdf", "page_number": 1},
            {"source": "travel.pdf", "page_number": 2},
        ]
        collection = Mock()
        collection.count.return_value = 2
        collection.get.return_value = {
            "ids": ["semantic", "lexical"],
            "documents": documents,
            "metadatas": metadatas,
        }
        collection.query.return_value = {
            "ids": [["semantic", "lexical"]],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [[0.2, 1.0]],
        }
        get_collection.return_value = collection

        hits = vector_store.query(
            "What is the customer name in the confirmation voucher?", n_results=1, tenant_id=7
        )

        self.assertEqual(len(hits), 1)
        self.assertIn("CUSTOMER NAME: Avery SamplePerson", hits[0]["text"])
        self.assertEqual(hits[0]["metadata"]["page_number"], 2)
        self.assertEqual(collection.query.call_args.kwargs["n_results"], 8)

    def test_missing_or_conflicting_field_does_not_guess(self):
        missing = self.make_hit({"customer_name": ""}, "CUSTOMER NAME AGENT MOBILE NO EMAIL")
        unclear = (
            "I couldn't reliably identify the customer name in this document. "
            "Please check the source passage."
        )
        self.assertEqual(answer_from_document("What is the customer name?", [missing]), unclear)
        conflicting = [
            self.make_hit({"customer_name": "Avery SamplePerson"}),
            self.make_hit({"customer_name": "Jordan Otherperson"}),
        ]
        self.assertEqual(answer_from_document("What is the customer name?", conflicting), unclear)
        ambiguous_dates = self.make_hit(
            {"booking_date": "17 Sep 2026", "journey_date": "05 Oct 2026"}
        )
        self.assertIn(
            "multiple possible fields",
            answer_from_document("What is the date?", [ambiguous_dates]),
        )

    def test_general_question_returns_a_short_evidence_snippet(self):
        passage = (
            "Confirmation voucher package details begin here. "
            + "Background information not related to the question. " * 20
            + "The trip includes a ferry transfer to Havelock Island. "
            + "Additional voucher conditions follow. " * 20
        )
        answer = answer_from_document("What does the trip include?", [self.make_hit({}, passage)])
        self.assertIn("ferry transfer to Havelock Island", answer)
        self.assertLessEqual(len(answer), 260)
        self.assertIn("page_number", self.make_hit({}, passage)["metadata"])


if __name__ == "__main__":
    unittest.main()
