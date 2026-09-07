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