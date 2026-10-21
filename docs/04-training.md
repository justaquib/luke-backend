# Training Guide — FAQ Learning Flow

The assistant can be trained in three ways, all from the **Trainer** page (`/trainer` in Angular) or via the REST API.

## Flow overview

```
user asks question
        │
        ▼
  /query  ─── 1. search "faq" Chroma collection
        │        └─ if distance ≤ FAQ_MATCH_THRESHOLD (default 0.55)
        │           → return trained answer (source: "faq")
        │        └─ otherwise
        ▼
  2. auto-intent: search FAQ collection loosely
        │        └─ if FAQ_MATCH_THRESHOLD < distance ≤ INTENT_MATCH_THRESHOLD
        │           → return that FAQ's answer (source: "intent")
        │        └─ otherwise
        ▼
  3. search "docs" Chroma collection
        │        └─ if hits → return top chunk
        │        └─ otherwise
        ▼
  4. log to pending_questions, return "A trainer has been notified"
```

**Auto-intent** lets the AI match paraphrased questions (e.g. "what's the leave policy?" → "How many vacation days do employees get?") to an existing FAQ without retraining. The matched canonical question and distance are surfaced in `QueryResponse.intent` and the chat UI shows them as an `Intent match` source tag.

## Stages of auto-intent

The implementation is staged:

1. **Stage 1 — Auto-route to closest FAQ variant.** Live now. Whenever a question is a bit too far from any approved FAQ but still within `INTENT_MATCH_THRESHOLD` (default `0.75`), the closest FAQ's answer is returned as `source: "intent"`. The frontend marks these as `Intent match` and shows the canonical question it matched.
2. **Stage 2 — Manual merge from the Trainer Inbox.** Live now. The Inbox row exposes a **Merge duplicates** button that fetches a preview of pending questions within `INTENT_MERGE_THRESHOLD` (default `0.35`) and lets the trainer select which ones to group under one canonical FAQ + answer.
3. **Stage 3 — Named intents.** Foundation only. A new `intents` table and `intents` Chroma collection back future classifier work. CRUD endpoints are already exposed at `/intents`, but they are not yet used by `/query`.

## Option 1 — Review the inbox (recommended)

The most common loop:

1. Users ask questions. Some have no trained answer and land in `pending_questions`.
2. Open `/trainer` in the frontend.
3. The **Inbox** tab shows every unresolved question, sorted by how often it was asked.
4. Type the correct answer next to each one and click **Save & train**.
5. The Q↔A is moved into the `faqs` table, the same question is upserted into the `faq` Chroma collection, and the next time anyone asks it, the trained answer is returned.

API equivalents:

```bash
# list pending
curl http://localhost:8000/faq/inbox

# answer pending id=1
curl -X POST http://localhost:8000/faq/inbox/1/answer \
  -H "Content-Type: application/json" \
  -d '{"answer":"The right answer…"}'
```

## Option 2 — Add a Q↔A directly

Use the **+ New** tab in the Trainer, or:

```bash
curl -X POST http://localhost:8000/faq \
  -H "Content-Type: application/json" \
  -d '{"question":"What are office hours?","answer":"Mon–Fri 9am–6pm."}'
```

## Option 3 — Edit or delete an existing answer

```bash
# edit
curl -X PATCH http://localhost:8000/faq/3 \
  -H "Content-Type: application/json" \
  -d '{"answer":"Updated answer."}'

# delete
curl -X DELETE http://localhost:8000/faq/3
```

Edits and deletes are mirrored in the Chroma `faq` collection automatically.

## How "did this match" is decided

ChromaDB returns a **distance** for each candidate (cosine, lower = closer).
`FAQ_MATCH_THRESHOLD` (default `0.55`, set in `.env`) is the cutoff.

- `distance ≤ 0.55` → trained answer is returned.
- `distance > 0.55` → falls through to documents, then to the pending inbox.

If your trained answers are not being matched, raise the threshold (e.g. `0.7`). If irrelevant answers are sneaking in, lower it (e.g. `0.4`).

## Training the brain

The brain is the writing model behind the assistant. It is trained from
everything the tenant has given, not only from the lessons in the learner:

| Source | What it is |
| --- | --- |
| Lessons | The question and the answer of every lesson, and the starter lessons of the grades the tenant reached. |
| Approved answers | The questions and the answers that were approved in the Trainer. |
| Corrections | The answers the trainer corrected. |
| Documents | The chunks of every uploaded document. |
| Words | The words and the meanings that were taught with the dictionary. |
| Sentences | The sentences that were taught on their own. |

Every source becomes a sentence, so its words reach the vocabulary, and a
request with its answer, so the model learns to answer a request it was never
worded for. The questions of the approved answers are learned that way too,
so the trainer's own answers are written in the model's words and not only
looked up.

Training takes about a minute and runs in its own process, so the app stays
usable while the brain learns. The brain page asks the brain again until the
training is done, and the small info button beside **Train the brain** opens
a panel that says what the brain has trained from and what the training does
with it.

```bash
# start the training
curl -X POST http://localhost:8000/language-model/brain/train \
  -H "Authorization: Bearer $TOKEN"

# read what it trains from
curl http://localhost:8000/language-model/brain/sources \
  -H "Authorization: Bearer $TOKEN"
```

## Correcting an answer that was wrong

Mark an answer wrong in the chat and write the answer that should have been
given. The correction is taught straight away:

- the right answer is stored as a trained answer, so the assistant answers
  the question correctly from then on,
- the correction is kept in the brain's memory, so the writing model learns
  the wording and the heads learn from it,
- the question is asked again to check that the assistant keeps it.

```bash
curl -X POST http://localhost:8000/language-model/brain/correction \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"What are the rudiments?","answer":"I do not know.",
       "correction":"The rudiments are the basic first principles of a subject."}'
```

## Data model

**Postgres**

- `faqs(id, question, answer, created_at, updated_at)` — one row per trained Q↔A.
- `pending_questions(id, question, occurrences, resolved, created_at, last_seen_at)` — every question that the bot could not answer.

**ChromaDB**

- `faq` collection — `id` = `faqs.id`, `document` = question, `metadata.answer` = answer.
- `docs` collection — unchanged, holds your raw document chunks.

## Tips

- Phrase the question the way real users ask. The match is by embedding similarity, not keyword.
- Short, unambiguous answers work best.
- If the same question is asked in many ways, add multiple FAQs.
- The inbox counter `occurrences` shows the most-asked unanswered question first — answer those first for the biggest payoff.