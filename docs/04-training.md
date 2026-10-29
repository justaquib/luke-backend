# Training Guide — FAQ Learning Flow

The assistant can be trained through the **Training room** page (`/dashboard/trainer` in Angular), the brain page (`/dashboard/brain`), or the REST API.

## Flow overview

```
user asks question
        │
        ▼
  /query  ─── 1. search "faq" Chroma collection
        │        └─ if distance ≤ FAQ_MATCH_THRESHOLD (default 0.28)
        │           → return trained answer (source: "faq")
        │        └─ otherwise
        ▼
  2. auto-intent: search FAQ collection loosely
        │        └─ if FAQ_MATCH_THRESHOLD < distance ≤ INTENT_MATCH_THRESHOLD
        │           → return that FAQ's answer (source: "intent")
        │        └─ otherwise
        ▼
  3. answer from the taught material (conversation, lessons, learner facts)
        │        └─ a greeting, the model's name, thanks or a goodbye
        │           → return the conversation answer (source: "assistant_profile")
        │        └─ otherwise the learner's own answer (source: its lesson kind)
        │        └─ otherwise
        ▼
  4. live sources: weather, then Wikipedia (when enabled for the tenant)
        │        └─ if an enabled source answers → return it
        │        └─ otherwise
        ▼
  5. search the "docs" Chroma collection
        │        └─ if hits → return top chunk
        │        └─ otherwise
        ▼
  6. log to pending_questions, return "A trainer has been notified"
```

**Auto-intent** lets the AI match paraphrased questions (e.g. "what's the leave policy?" → "How many vacation days do employees get?") to an existing FAQ without retraining. The matched canonical question and distance are surfaced in `QueryResponse.intent` and the chat UI shows them as an `Intent match` source tag.

## Stages of auto-intent

The implementation is staged:

1. **Stage 1 — Auto-route to closest FAQ variant.** Live now. Whenever a question is a bit too far from any approved FAQ but still within `INTENT_MATCH_THRESHOLD` (default `0.36`), the closest FAQ's answer is returned as `source: "intent"`. The frontend marks these as `Intent match` and shows the canonical question it matched.
2. **Stage 2 — Manual merge from the Trainer Inbox.** Live now. The Inbox row exposes a **Merge duplicates** button that fetches a preview of pending questions within `INTENT_MERGE_THRESHOLD` (default `0.35`) and lets the trainer select which ones to group under one canonical FAQ + answer.
3. **Stage 3 — Named intents.** Foundation only. A new `intents` table and `intents` Chroma collection back future classifier work. CRUD endpoints are already exposed at `/intents`, but they are not yet used by `/query`.

## Option 1 — Review the inbox (recommended)

The most common loop:

1. Users ask questions. Some have no trained answer and land in `pending_questions`.
2. Open `/dashboard/trainer` in the frontend.
3. The **Inbox** tab shows every unresolved question, sorted by how often it was asked.
4. Type the correct answer next to each one and click **Save & train**.
5. The Q↔A is moved into the `faqs` table, the same question is upserted into the `faq` Chroma collection, and the next time anyone asks it, the trained answer is returned.

An answer that contradicts one already approved goes to the same inbox instead of
silently replacing it. `contradictions.flag_conflict` runs whenever a FAQ is
created, edited or answered: it looks for an approved FAQ whose question is the
same set of words (order and punctuation ignored) and whose answer shares less
than `SAME_ANSWER_SIMILARITY` (0.8) of its words with the new one — the Jaccard
overlap of the two word sets. A rephrasing (0.8 or above) is not a conflict and
nothing is raised. A conflict adds a `pending_questions` row whose `note` holds
both answers, "New answer: … — approved answer: …. Choose the answer that
stands.", and the inbox shows it beside the question so the trainer picks. The
approved answer keeps serving until they do.

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
`FAQ_MATCH_THRESHOLD` (default `0.28`, set in `.env`) is the cutoff.

- `distance ≤ 0.28` → trained answer is returned.
- `0.28 < distance ≤ 0.36` → the answer is returned with a "not sure" note (`source: "intent"`).
- `distance > 0.36` → falls through to documents, then to the pending inbox.

The values were calibrated on 2026-10-27 against the cosine index: one positive
and 17 negatives, the lowest negative at `0.4455`, which is why both limits sit
below it. Re-run `scripts/calibrate_thresholds.py` when more feedback exists.

If your trained answers are not being matched, raise the threshold a little
(e.g. `0.30`). If irrelevant answers are sneaking in, lower it (e.g. `0.22`).
On the cosine scale a step of `0.01` is a real change; the old L2 numbers
(`0.55`/`0.75`) are twice as large for the same angle.

## What an upload does before it is stored

An upload is accepted and read later: `POST /upload` checks the type, the size
(`MAX_UPLOAD_BYTES`, 25 MB by default) and that the bytes match the name, refuses
content whose SHA-256 the workspace already holds, writes the file and answers
**202** with a job id. A pool of `INGEST_WORKERS` (2) background workers then does
the reading, one file at a time each, and the page follows the job over
`GET /ingest/jobs/{id}/events`.

Before a single passage is stored, four things happen to it:

- **The furniture goes out.** `quality/boilerplate.py` drops lines that repeat
  across pages (running headers and footers), bare page numbers and copyright
  lines, so a hundred copies of "Confidential — page 7 of 40" never become a
  hundred training passages.
- **Personal detail, if you ask.** `PII_SCRUB_ENABLED` is off by default. When it
  is on, emails, phone numbers and long identifiers are masked before chunking, so
  the masked form — never the original — is what the index and the trainer see.
- **Each passage gets a quality score.** `documents.quality` is
  `language factor × text density × (1 − repetition)`: 1.0 for confirmed-English
  prose, 0.5 when the language cannot be confirmed, and 0.0 for a fragment under
  the shortest learnable sentence or a passage that repeats itself 60% or more.
  The score orders the passages a training run prefers; it does not delete them.
  Passages stored before the score existed are measured by
  `scripts/migrate_quality.py`, never by a rewrite at request time.
- **Repeats collapse.** A passage whose SHA-256 is already in the workspace is an
  exact duplicate, and one whose 64-bit SimHash is within five bits of a passage
  the workspace holds — or of an earlier passage in the same file — is a near
  duplicate. Neither is stored or embedded; both are counted, and the counts
  (`exact_duplicates`, `near_duplicates`) are what the job card shows.

A job the trainer stops, or one that fails, leaves nothing behind: the vectors it
had written are removed, no `documents` rows stay, and the file remains on disk so
**Retry** can read it again from the start. A job interrupted by a restart is
marked retryable when the app boots.

## Training the brain

The brain is the word model behind the learner's sentences. It is trained from
everything the tenant has given, not only from the lessons in the learner:

| Source | What it is |
| --- | --- |
| Lessons | The question and the answer of every lesson, and the starter lessons of the grades the tenant reached. |
| Conversations | The greetings, the model's name, thanks, goodbyes and small talk; the pairs are built from the name saved in Settings, and the chat answers from the same pairs directly. Every expression taught on `/dashboard/expressions` (open, close, pause, resume) joins this source too. |
| Approved answers | The questions and the answers that were approved in the Trainer. |
| Corrections | The answers the trainer corrected. |
| Documents | The chunks of every uploaded document, cut into 3–50-word passages and kept by extraction quality inside the training budget. |
| Words | The words and the meanings that were taught with the dictionary. |
| Sentences | The sentences that were taught on their own. |

Every source becomes a sentence, so its words reach the vocabulary. The
answers of the lessons, approvals, corrections and conversations are collected
in the wordings they can be asked for, so the words a trainer would use for a
subject are in the corpus too. The chat answers from the taught material as it
was taught. The word model does not write the chat answer; when nothing is
taught, the read-only foundation checkpoint writes one (see below), and that
answer never becomes training material.

The model's name and starting version are set once, under **Model identity** in
`/dashboard/settings` (`GET`/`PUT /language-model/settings`). The name is not
only a label: it is used in the conversation pairs, so renaming the model
changes how it greets and introduces itself, and the next training learns the
new name. The starting version is the floor of the level ladder the brain page
shows.

Training runs in its own process, so the app stays usable while the brain
learns. How long it takes is measured, not fixed: the word model makes eight
passes over the examples, capped by a wall-clock budget. The run is read on its
held-out set, not on a single batch, and stops at the end of its pass or when
the held-out read stops improving. The brain page asks again until the
training is done and shows the checklist, the steps and the time left; the
small info button beside **Train** opens a panel that says what the brain has
trained from and what the training does with it.

### The tokenizer a training uses

The tokenizer is part of the model, not a setting shared between checkpoints.
A new training uses the word tokenizer by default, and the same corpus and
configuration always rebuild it. Byte-level BPE is available as an explicit
opt-in (`tokenizer_kind: "bpe"`, with `vocabulary_size`); it is measured and
tested but is **not** the default, because on the current 136K-character corpus
the word tokenizer scored 1.3298 held-out bits per character against BPE's
1.4807 at 268,288 fewer parameters (`docs/07-model-measurements.md`).

- A BPE vocabulary is built from the training side of the group split only, so
  a held-out sentence can never shape it. The record keeps the sentence and
  character counts of both sides.
- A schema-3 checkpoint records `tokenizer: {kind, id, vocab_size,
  special_ids}`. A checkpoint from before schema 3 has no record and loads
  through the word adapter with exactly its old ids.
- A BPE artifact is stored under `.language_model/tokenizers/{sha256}/` and
  verified before the weights are used. A missing, changed or mismatched
  artifact is refused with an actionable error; nothing is remapped or
  truncated.
- Luke 0.0.1 and every fine-tuned teacher keep their own tokenizer files. A
  scratch tokenizer is never loaded for them, and theirs is never replaced.

```bash
# measure the tokenizer candidates against the word baseline (read-only)
cd assistance_back && .venv/bin/python scripts/measure_tokenizer.py --dry-run
.venv/bin/python scripts/measure_tokenizer.py --tenant 0
```


```bash
# start the training
curl -X POST http://localhost:8000/language-model/brain/train \
  -H "Authorization: Bearer $TOKEN"

# read what it trains from
curl http://localhost:8000/language-model/brain/sources \
  -H "Authorization: Bearer $TOKEN"
```

## Taught answers and written answers

Luke 0.0.1 has two kinds of answer, and they are never mixed:

- **Taught** — a saved answer, an approved answer, a lesson, a dictionary entry
  or a taught expression. It is served as the trainer wrote it, and it always
  wins. The chat labels it `retrieved`.
- **Written** — when every taught source misses, the pinned
  Qwen2.5-0.5B-Instruct checkpoint writes an answer. It is labelled
  `generated`, marked unverified, carries no citation, and is **never stored**
  as a lesson, FAQ, correction or training pair. It adds wording to a gap; it
  does not add knowledge.
- **Composed** — an opt-in answer written from numbered retrieved passages.
  The model must cite them and the verifier checks every citation and every
  number and name against the passages; when the check fails the passages
  themselves are served. The chat labels it `hybrid`.

The checkpoint is read-only. Training never touches its weights, and the
package export never includes it as Luke's own knowledge. The trainer teaches
openings, closings, pauses and resumes on `/dashboard/expressions`; those
expressions are answered directly and also join the conversation training
pairs, so the taught voice is the one that is learned.



The **Base model** tab in the training room fine-tunes a small Hugging Face
causal language model on this tenant's material. It was retired on 2026-10-25
with the from-scratch answer writer and restored on 2026-10-26: the trainer
asked for it back, and the chat now lets them answer with Luke or with any
teacher saved on this machine.

- `GET /language-model/huggingface` reports the measured machine (device,
  `torch`, `transformers`), the offered base models, the saved teachers and the
  last run.
- `POST /language-model/huggingface/train` starts one run in a background
  thread (`model_id`, `steps`); a second run is refused with `409` while one is
  going. The run records `download` and `tune` stages, the steps done, an ETA,
  the loss it measured and the material it used, under
  `hf-progress-<tenant>.json`; `last-run.json` keeps the finished report.
- `DELETE /language-model/huggingface/teachers/{folder}` removes one teacher's
  weights after a confirmation. The last-run report and the answers it already
  wrote stay; a folder name outside the teacher folder is rejected.
- `GET /language-model/models` lists Luke and every installed teacher, and
  `POST /language-model/predict` returns the next words **and** the attention
  the selected model measured for a prefix. Luke reads from the retained word
  model; a teacher reads with `attn_implementation="eager"`.

Weights live beside the tenant's learner data, so the material and the teacher
stay together:

- `.language_model/huggingface/tenant-<id>/teacher-<model>/` per teacher,
- `last-run.json` in the tenant folder, `hf-progress-<tenant>.json` in the
  learner's folder.

To reclaim the space, stop the app and delete a teacher folder by hand (or use
the trash control on the tab). Training the word model does not need it.

## Moving the teacher's knowledge into Luke

A teacher knows this tenant's material in its own words, but only while its
weights are loaded. **Transfer into Luke** asks the teacher every question the
material can still ask and keeps the answers as trained answers, so the
knowledge stays when the weights are removed.

- `GET /language-model/huggingface/transfer` reports the installed teachers,
  how many questions are available (by source), a sample of them, the run in
  progress and the last report. A question the tenant already has — at any
  status — is not asked again.
- `POST /language-model/huggingface/transfer` (`folder`, `limit` 1–200, default
  40) loads the teacher and asks the questions in a background thread. The
  sources are lessons, starter lessons, corrections, taught words, documents
  and one topic question per subject. The report is written step by step under
  `.language_model/learner-<tenant>/hf-transfer-<tenant>.json`.
- An answer is kept only when it stays grounded in the reference it was given:
  a short reference needs one shared content word, a longer one needs two and
  a third of the answer's content words. An ungrounded answer is counted as
  skipped, never stored.
- A kept answer is stored as an approved FAQ and indexed at once, so Luke
  answers it without the teacher loaded.
- `POST /language-model/huggingface/transfer/undo` removes exactly the answers
  the last transfer created (their ids are in its report) and nothing else.
- One transfer runs at a time. It is refused while a fine-tune is going, and a
  fine-tune or a teacher removal is refused while a transfer is reading the
  teacher.

## Downloading Luke as one package

**Download Luke** on the Base model tab writes one zip (format 2) with the word
model Luke answers with, the tokenizers, the learner state, the indexed
material and, in full mode, the base checkpoint itself.

- `GET /language-model/export/preview` counts the material by source, lists the
  checkpoints, the unresolved questions and both modes with their size and
  file count, without building anything.
- `GET /language-model/export` writes the zip to a temporary file, sends it with
  the download file name, and removes the file once the response is sent. The
  mode is `slim` or `full`: **slim** carries the base checkpoint as a reference
  (name, revision, licence, weight SHA-256) and **full** also carries its nine
  weight files, so a fresh machine can answer without downloading anything.
- The zip holds `manifest.json`, `README.md`, `weights/checkpoints/`,
  `weights/tokenizers/`, `weights/base/` (full only), `state/`,
  `dataset/*.jsonl` and `raw/`. The manifest records the checkpoint and
  tokenizer identities, the licence, the revisions, the embedding model, the
  collection settings, the counts, and the size and SHA-256 of every file.
- Teacher weights and the retired writer are left out; the teacher's knowledge
  reaches Luke through transfer instead. Document files are kept under `raw/`
  because their rows point at them, so a restore can re-index the same text.
- The package is scoped like the rest of the API: a tenant gets only its own
  rows and files, and the administrator gets only the rows and files that have
  no tenant. A package is refused by any other workspace.

**Restore a package** with `POST /language-model/package/import` and the zip as
the uploaded file. It defaults to a dry run (`dry_run=true`) that checks the
manifest, every file hash, the counts, the row types and the local references
and reports what it would do, without touching a row, a file or a collection.
Applying it needs `dry_run=false`, and a workspace that already holds rows also
needs `replace=true`. The replace is transactional: the old rows, files and
collections are put back if any check fails. After the rows are written the
index is rebuilt from the database with the embedding and collection settings
the manifest names, and the counts, one document and one FAQ are read back
before the import reports success. The measured sizes of both modes and a real
dry run are in `docs/07-model-measurements.md`.

## Correcting an answer that was wrong

Mark an answer wrong in the chat and write the answer that should have been
given. The correction is taught straight away:

- the right answer is stored as a trained answer, so the assistant answers
  the question correctly from then on,
- the correction is kept in the brain's memory, so the heads learn from it,
- the question is asked again to check that the assistant keeps it.

```bash
curl -X POST http://localhost:8000/language-model/brain/correction \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"What are the rudiments?","answer":"I do not know.",
       "correction":"The rudiments are the basic first principles of a subject."}'
```

## Web pages are never auto-trained

The web evidence step (Phase 4) can quote an allowlisted page, but it never
trains from one:

- An allowlisted page is indexed into the `web` collection as unverified
  evidence and cited with its address and fetch date. It is not copied into
  `faqs`, the lesson state or any training pair.
- A search result outside the allowlist becomes a `pending` FAQ candidate
  whose question is `"<page title> — <address>"` and whose answer is the first
  900 characters of the text. It is not indexed, so no FAQ or intent lookup can
  serve it, and it waits for a trainer exactly like an unanswered question.
- A trainer who wants the page to answer must teach it as a lesson or approve
  a FAQ themselves; the taught answer then wins over the web step, because the
  taught steps run first.
- Refresh only re-reads pages; a changed page supersedes the old snapshot and
  is re-indexed as unverified evidence. It does not add training material.

## Data model

**Postgres**

- `faqs(id, question, answer, created_at, updated_at)` — one row per trained Q↔A.
- `pending_questions(id, question, note, occurrences, resolved, created_at, last_seen_at)` — every question the bot could not answer, and every answer that contradicts an approved one (`note` holds both answers).
- `uploaded_files(id, source, content_hash, size_bytes, tenant_id, created_at)` — one row per file kept on disk.
- `ingest_jobs(id, source, status, step, progress, done, total, chunks, exact_duplicates, near_duplicates, error, tenant_id, created_at, started_at, finished_at)` — the background read of one upload.
- `documents(id, source, chunk_index, text, tenant_id, created_at, document_id, char_start, char_end, page, content_hash, simhash, language, quality)` — one row per stored passage, with its provenance and its quality score.

**ChromaDB**

- `faq` collection — `id` = `faqs.id`, `document` = question, `metadata.answer` = answer.
- `intents` collection — `id` = `intents.id`, `document` = the intent name.
- `docs` collection — one vector per stored passage, with source, workspace, page,
  character span, content hash, language and quality in its metadata.
- `web` collection — passages of allowlisted pages, marked unverified.

## Tips

- Phrase the question the way real users ask. The match is by embedding similarity, not keyword.
- Short, unambiguous answers work best.
- If the same question is asked in many ways, add multiple FAQs.
- The inbox counter `occurrences` shows the most-asked unanswered question first — answer those first for the biggest payoff.