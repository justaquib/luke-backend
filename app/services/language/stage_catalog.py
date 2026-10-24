"""The static stage ladder of the learner curriculum."""

from __future__ import annotations

STAGES = [
    {
        "id": "nursery",
        "title": "Nursery",
        "description": (
            "First ideas: letters, numbers, colors, shapes, feelings, greetings, "
            "and the world around us."
        ),
        "subjects": [
            "Language",
            "Numbers",
            "Shapes",
            "Colors",
            "Feelings & expressions",
            "My World",
            "Punctuation",
        ],
    },
    {
        "id": "lkg",
        "title": "Lower Kindergarten (LKG)",
        "description": "Build early reading, counting, drawing, and everyday knowledge.",
        "subjects": [
            "Language",
            "Math",
            "Shapes",
            "Nature",
            "Daily Life",
            "English Grammar",
            "Sentence Making",
            "Tenses",
            "Calculation",
        ],
    },
    {
        "id": "ukg",
        "title": "Upper Kindergarten (UKG)",
        "description": "Practise simple reading, number operations, patterns, and observation.",
        "subjects": [
            "Language",
            "Math",
            "Patterns",
            "Science",
            "Social Skills",
            "English Grammar",
            "Sentence Making",
            "Calculation",
            "Numbers",
        ],
    },
    *[
        {
            "id": f"class_{grade}",
            "title": f"Class {grade}",
            "description": f"Build school knowledge and lessons for Class {grade}.",
            "subjects": ["Language", "Math", "Science", "Social Studies", "Arts"],
        }
        for grade in range(1, 13)
    ],
    {
        "id": "graduation",
        "title": "Graduation",
        "description": "Add undergraduate-level lessons by subject and course.",
        "subjects": ["Course", "Subject", "Research", "Professional Skills"],
    },
]
GRADE_IDS = {item["id"] for item in STAGES}
LEGACY_STAGES = {"letters", "words", "sentences", "lessons"}
