# Project Structure & Architecture

A walkthrough of every file in the project and how the pieces fit together.

## Top-level layout

```
assistance/
├── app/                      Python source code
│   ├── main.py               FastAPI app + routes
│   ├── config.py             Pydantic settings (loads .env)
│   ├── schemas.py            Request / response models
│   ├── db/
│   │   └── database.py       SQLAlchemy engine + Document model
│   └── services/
│       ├── nlp.py            spaCy chunking + entity extraction
│       ├── vector_store.py   ChromaDB persistent client
│       └── ingestion.py      Ingest text -> chunk -> embed -> store
├── docs/                     Step-by-step guides
│   ├── 01-setup.md           Python project setup
│   ├── 02-frontend-connection.md  Angular wiring
│   └── 03-architecture.md    This file
├── samples/                  Example training data
├── start                     one-command backend launcher
├── scripts/                  startup, structure check, measurements
├── docker-compose.yml        PostgreSQL 16
├── requirements.txt          Python dependencies
├── .env.example              Environment template
├── .env                      (not committed) your local config
├── .gitignore
└── README.md                 Quick-start + curl examples
```

## Runtime architecture

```
                  ┌──────────────────────────────────────┐
                  │            Angular App               │
                  │  (your existing project, port 4200)  │
                  └──────────────┬───────────────────────┘
                                 │ HTTP (CORS)
                                 ▼
                  ┌──────────────────────────────────────┐
                  │  FastAPI (uvicorn, port 8000)        │
                  │  - /ingest    POST                   │
                  │  - /query     POST                   │
                  │  - /stats     GET                    │
                  │  - /health    GET                    │
                  └────┬─────────────────────────┬───────┘
                       │                         │
            chunks + vectors                metadata rows
                       ▼                         ▼
            ┌────────────────────┐    ┌────────────────────┐
            │  ChromaDB (local)  │    │  PostgreSQL 16     │
            │  ./.chroma/        │    │  (Docker, :5433)   │
            │  - embeddings      │    │  - documents table │
            │  - similarity      │    │                    │
            └────────────────────┘    └────────────────────┘
                       ▲
                       │ sentence chunking + NER
                       │
            ┌────────────────────┐
            │  spaCy (in-process)│
            │  en_core_web_sm    │
            └────────────────────┘
```

## Request flow

### Ingest (`POST /ingest`)

1. Client sends `{source, text}` to FastAPI.
2. `ingest.ingest_text` calls `nlp.chunk_text(text)` which:
   - Cleans whitespace and keeps the line breaks the sentencizer reads.
   - Loads `en_core_web_sm` (cached after first use).
   - Adds the `sentencizer` pipe if no parser is present.
   - Splits into sentences and packs them into chunks (max 800 chars, 100-char overlap). A chunk never starts or ends in the middle of a word, a sentence longer than the limit is split between words, and a line that starts a new question (`Q.`, `Question`) starts a new chunk instead of finishing the previous one.
3. Each chunk is added to ChromaDB with a stable id like `notes-3` and a `{source}` metadata.
4. The same chunks are inserted as rows in the Postgres `documents` table for audit/search.
5. Response: `{source, chunks, start_id}`.

### Query (`POST /query`) and the answer stream (`POST /query/stream`)

1. Client sends `{question, n_results}`.
2. `nlu.read_question` reads the question once: its words, each word's lemma and part of speech, the meaning and up to two synonyms of every key word from the offline dictionary, the question type and the key words. A dictionary that cannot be opened leaves the reading empty; it never fails the request.
3. `answer_flow.iter_answer_events` emits the stages as they happen: `reading`, then `thinking` (the question type and the sources it plans to consult), then one `connecting` per source as it is consulted, and finally `answer`. `POST /query` consumes the same iterator, so the JSON and the stream cannot drift.
4. `vector_store.search_faq` looks for a trained answer in the `faq` collection. If the best match has a cosine distance ≤ `FAQ_MATCH_THRESHOLD` (default `0.55`), the trained answer is returned with `source: "faq"`.
5. Otherwise a loose FAQ match within `INTENT_MATCH_THRESHOLD` is returned with `source: "intent"`.
6. Otherwise the shared taught-answer path (`answering.answer_from_curriculum`) answers: conversation first (a greeting, the model's name from Settings, thanks, a goodbye, small talk; `source: "assistant_profile"`), then the learner's lessons and quick facts. This is the same path the learner page uses, so a question cannot be answered one way in the chat and another way in the learner.
7. Otherwise the live sources answer when they are enabled: weather, then Wikipedia.
8. Otherwise `vector_store.query` falls back to the `docs` collection. It fetches twice as many semantic candidates as it returns, then `rerank_hits` orders them by a weighted blend of the embedding distance (0.4) and the share of the question's meaningful terms the chunk names (0.6). Stopwords count for neither, so a shared "the" cannot lift a passage.
9. If nothing answers, the question is logged to `pending_questions` (occurrence counter incremented if it was already there) and a "trainer has been notified" placeholder is returned with `source: "pending"`.
10. Every answer is finished once (`language/finishing.py`): outer whitespace trimmed, the first letter capitalized, and a full stop added only when the text ends in a letter or digit. Words are never reworded. The answer carries an optional `understanding` (words read, plan, connections, grammar notes), and `nlp.extract_entities` runs spaCy NER over the question for the frontend to display.
11. Response: `{question, source, answer, hits, entities, understanding, pending_id?}`.

`POST /query/stream` returns the same answer as a server-sent event stream: `reading`, `thinking`, one `connecting` per source, `answer` (the full JSON above), then `done`. An `error` frame is terminal and no `done` follows it. The caller's token is checked before the stream opens, so 401/403 stay normal HTTP responses.

### Reading and connecting a question

`app/services/nlu/` reads the question before anything is looked up. One spaCy parse gives each word its lemma and part of speech (the blank pipeline still works), `reasoning.entities.entities_of` picks the key words, `dictionary.lookup_word` gives the first definition of each key word and `dictionary.search_expansions` up to two synonyms, and `reasoning.intent.shape_kind` names the question type.

`app/services/answer_flow.py` owns the decision path. Each finder returns a `SourceStep` whose `Connection` records the source, an outcome (`match`, `miss`, `pass`, `fail` or `skipped`), the measured detail (a distance, the lesson evidence, "not enabled", "no matching page") and the question that matched. A stage is never invented: the page shows exactly the connections the backend produced, and the same list is stored with the answer and shown in its details.

### Stats (`GET /stats`) and `/health`

Trivial helpers for the frontend to know if the backend is alive and how much has been ingested.

### The language learner (`/language-model`)

The learner and the brain pages read and write the tenant's own small model. The
routes live in `app/routers/language_model.py` (the learner) and
`app/routers/language_brain.py` (the brain); they only check the caller and
read the request; the work is done by the services in `app/services/language/`.

| Route                                  | What it does                                              |
| -------------------------------------- | --------------------------------------------------------- |
| `GET /language-model/state`            | The saved state: stages, lessons, words, active test      |
| `GET /language-model/store`            | What the store holds: the files, the numbers, the size on disk |
| `GET /language-model/words`            | The words the answering model can write with, page by page |
| `GET /language-model/graph`            | The shape of what the tenant has learned                   |
| `GET /language-model/brain`            | What the brain knows about itself, and the level it reached |
| `GET /language-model/brain/sources`    | What the brain trains from, and how the training works    |
| `GET /language-model/settings`         | The model's name and the version it starts from            |
| `PUT /language-model/settings`         | Save the model's name and starting version                 |
| `POST /language-model/brain/correction` | Teach the right answer to a question answered wrongly    |
| `POST /language-model/brain/train`     | Train the brain on the tenant's own material               |
| `POST /language-model/lessons/check`   | Report what the learner already knows about a lesson       |
| `GET /language-model/lessons/duplicates` | The lessons that repeat each other, grouped for review    |
| `POST /language-model/lessons/duplicates/accept` | Keep every lesson of a group: the repeat is on purpose |
| `POST /language-model/lessons`         | Teach a lesson, or a story, a lesson or a topic            |
| `PUT /language-model/lessons/{id}`     | Edit a lesson the learner already holds                    |
| `DELETE /language-model/lessons/{id}`  | Remove a lesson                                            |
| `POST /language-model/preview`         | Answer a question with the learner's own material only     |
| `POST /language-model/preview/stream`  | The same answer, read and connected stage by stage         |
| `POST /language-model/test/start`      | Start a mock test, a final exam, or a trainer led test     |
| `POST /language-model/test/question`   | Ask the learner a question during a grade test             |
| `POST /language-model/test/score`      | Score the learner's last answer                            |
| `POST /language-model/test/finish`     | Close the test and open the next grade when it is passed   |
| `GET /language-model/dictionary`       | Look a word up in the offline dictionary                    |
| `POST /language-model/dictionary/learn`| Teach the learner a word and the meaning it should keep    |
| `POST /language-model/sentence`        | Build a sentence from the words the learner knows          |
| `POST /parenting/chat/session`          | Start a coaching session between the parent and the coach |
| `GET /parenting/chat/session/{id}`      | Read one coaching session                               |
| `POST /parenting/chat/session/{id}/turn` | The coach answers the parent's turn from the taught lessons |
| `POST /parenting/chat/session/{id}/turn/stream` | The same turn, read and connected stage by stage     |

Every route needs the same bearer token as the rest of the API and only ever
reads or writes the tenant of the signed in user.

### The model choice and the Hugging Face teachers

`app/routers/huggingface.py` is thin; `app/services/model_choice.py` decides
which model answers a message and `app/services/huggingface/` owns the teachers
(catalogue, files, fine-tune, prompt and serving). A teacher is fine-tuned on
the tenant's own taught material, written under
`.language_model/huggingface/tenant-<id>/teacher-<model>/`, and loaded once and
kept while it answers.

| Route | What it does |
| --- | --- |
| `GET /language-model/models` | Luke and every installed teacher, with the default |
| `POST /language-model/predict` | The next words and the measured attention of the chosen model |
| `GET /language-model/huggingface` | The machine, the offered base models, the saved teachers and the last run |
| `POST /language-model/huggingface/train` | Fine-tune one base model in the background (one run at a time) |
| `DELETE /language-model/huggingface/teachers/{folder}` | Remove one teacher's weights |

The chosen model is sent with `/query` and `/query/stream`. A teacher writes
its answer from the retrieved reference (the matched answer plus the retrieved
passages) and must share enough of the reference's words to be shown; when it
does not, the matched answer is served and the `model` connection records the
fallback. A teacher's measured limits are in `docs/07-model-measurements.md`;
the chat always labels the answer with the model that wrote it.

A session lasts a week. `POST /auth/refresh` gives a signed-in user a new
token, so the app renews the session while it is in use instead of asking the
user to sign in again every day.

A lesson is a question and an answer. A story, a lesson or a topic is taught
as one passage instead: it is filed under the name it was given, every
sentence of it is taught on its own as well, and the learner can ask for it
by that name or by its topic.

The model's identity is tenant state too: `model-<tenant>.json` keeps the name
(default `Luke`) and the starting version (default `0.0.0`). The name is the
one the assistant uses when it speaks about itself, and the starting version
is the floor of the displayed level ladder. The greeting pairs are built from
the same name, so a rename changes the voice and the training together.

The whole ladder is data, in `app/services/brain_roadmap.py`: 101 rungs from
`0.0.0` to `1.0.0`, grouped into ten bands and ending with the Generalist. A
rung is marked `open` only when `LUKE_LEVELS` holds a real test for it, so the
page can show the full road while measurement stays at the last tested level
(`0.1.0`). `GET /language-model/brain` returns every rung with its band and
open state; the page locks every rung that is not the next open one, and the
displayed level still moves only on the trainer's approval.

Training the brain takes its time, so it runs in its own process: the app
stays usable while the model learns. The tenant's learner file carries a
`training-<tenant>.json` marker while that runs, and the page asks the brain
again until the marker is gone. One checkpoint is written next to the model
(`words-<tenant>.pt` for next-word prediction), and the page picks it up as
soon as the file changes, so the vocabulary grows with every lesson that was
taught.

### The neural engine

`app/services/neural/` is the one place that owns the models. Nothing outside
the package imports `torch` or opens a checkpoint file:

| Module | What it owns |
| --- | --- |
| `engine.py` | `NeuralEngine`: the word task of one tenant, its training, its saved file and the numbers the pages read. `engine_for(tenant_id)` keeps one engine per tenant and reloads it when the checkpoint changes. |
| `words.py` | `WordTask`: plain next-word prediction over the tenant's sentences, with the vocabulary that is the one live token list. |
| `training.py` | The training loop: a group split, fixed train and held-out batches, warmup then cosine, early stopping on the held-out loss, a wall-clock budget, and the report the page reads. |
| `measurement.py` | The held-out split, the fixed reads, and the chance loss a model is compared against. |
| `checkpoint.py`, `store.py` | Schema v2 checkpoint files, written atomically, plus the short run history the builder shows. |
| `serving.py` | The word surface the language services use; it reports the backend and the quality numbers behind it. |

The models are mapped from disk rather than copied into memory: the checkpoint
is read with `mmap=True` and its weights are assigned into a model skeleton, so
the pages that only show numbers never build a model at all, and a request
pays only for the task it uses. One engine is kept at a time, because a local
install serves one tenant.

### The word model

The served model is the word model: plain next-word prediction over the
tenant's sentences, used by the learner's sentence feature. It replaced the old
trigram, and when a tenant has not trained one yet, the learner page says so
and produces no predictions instead of falling back to n-gram guesses. Its
checkpoint and its held-out loss stand on their own, so the page reports the
quality of the model that actually runs.

The answer is not written by a model. A taught answer, an approved answer, a
lesson or a dictionary entry is served as it was taught, so the chat always
answers from the trainer's material.


### What the brain trains from

The training is built from every source the trainer has given, and they are
collected in one place, `app/services/language/training_materials.py`:

| Source | What it is |
| --- | --- |
| Lessons | The question and the answer of every lesson in the learner, and the starter lessons of the grades it reached. |
| Conversations | The greetings, the model's name, thanks, goodbyes and small talk, built from the name in Settings. |
| Approved answers | The questions and the answers the trainer approved in the Trainer. |
| Corrections | The answers the trainer corrected, so the model stops making the same mistake. |
| Documents | The chunks of every uploaded document. |
| Words | The words and the meanings that were taught with the dictionary. |
| Sentences | The sentences that were taught on their own. |

Every source becomes a sentence, so the words of it reach the vocabulary. The
answers of the lessons, approvals, corrections and conversations are also
collected in the wordings they can be asked for, so the words a trainer would
use for a subject are in the corpus too. The word model learns from that text
alone.

`GET /language-model/brain/sources` reads the same material and reports it,
so the trainer can see what the brain learns from and how the training works.
The report and the training cannot drift apart, because they are built from
the same list.

### Finding the lesson that answers a question

The learner has to find the lesson it was taught, even when the question is
asked in other words. The lookup lives in
`app/services/language/matching.py` and reads the question in the words of
the lesson:

- the mark that names the kind of lesson (`GK:`, `Poem:`, `Story:`) is not a
  word of the question, so it does not count against the lesson,
- the endings of the words are left off, so "rudiment" finds the lesson that
  was taught as "rudiments",
- the numbers are kept, so "the square root of 25" is not answered with the
  lesson about 16,
- two lessons that share their words are told apart by the wording of the
  question, so "What does the head do?" is not answered with "What is the
  head?".

A definition question that names no lesson title can still be answered by a
lesson taught in other words. `app/services/language/fallback.py` asks the
offline dictionary for up to six WordNet synonyms of the question word and
looks each one up with the same lesson matching. The fallback answers only
when **exactly one** distinct lesson is reached and the question shares a
stemmed key word with that lesson's title or answer; a synonym that reaches
nothing, or two lessons, leaves the dictionary answer in place. Multi-word
synonyms are not tried, because a phrase matches too loosely to name a lesson.
The connection is recorded as `fallback` with `pass` or `fail`, so the reading
trace shows why a synonym did or did not place the lesson.

### Answering a request, not a question

A lesson is taught as a question and an answer, so a lookup finds it. A
request is different: "write a short poem about the rain" shares almost no
words with the lesson "Poem: Rain Rain Go Away", so no amount of question
matching finds it.

The brain answers requests in three steps.

| Step | Where | What happens |
| --- | --- | --- |
| Read the request | `app/services/language/shapes.py`, `requests.py` | The request is matched against the shapes it knows: a poem about a topic, a list of a topic, some facts about a topic. The topic is taken out of it. |
| Find the material | `app/services/language/topics.py`, `requests.py` | The topic index is counted over the lessons of the kind the request asks for, so a word shared by many fact lessons can still name the one poem. A word that many lessons *of the same kind* share names none of them, and the assistant refuses rather than choosing one at random. |
| Serve the lesson | `app/services/language/requests.py` | The lesson the request named is returned as it was taught, marked `taught`. The words are the trainer's, never rewritten. |

A request about something never taught is never answered with an invented
answer: the material boundary returns no answer and the chat answers that a
trainer has been notified. Exact matches keep their order — FAQ, intent,
taught material, weather, wiki, documents — and the first one that has an
answer wins.

### The system administrator in the chat

The system administrator has no tenant of their own: they look after every
tenant instead of belonging to one. The chat answers them anyway, reading the
material of the tenant-less bucket (where the lessons they taught themselves
live) and, with no tenant filter on it, the approved answers and the
documents of every tenant they look after. Their own uploads and feedback land
in that tenant-less bucket (`tenant_id NULL`), so the material they add is
platform-owned rather than filed under a tenant. A user without a tenant is
still turned away, because they have no assistant of their own.

### Correcting an answer

`POST /language-model/brain/correction` takes the question, the answer that
was given and the answer that should have been given. It:

- stores the right answer as a trained answer of the tenant, so the next
  question of that kind is answered correctly from the FAQ collection,
- keeps the correction in the brain's own memory, so the heads learn from it,
- asks the question again and checks that the assistant keeps the correction,
  so the trainer is told whether it really learned it.

### The level test

`GET /language-model/brain?force=true` measures the rungs that have a real test
and keeps the result. The measure uses the generality probes: held-out
questions the training never showed, so a model that only memorises scores
nothing on them. The score is in `model_level.generality`, the probes are in
`model_level.generality_probes`, and the level itself still moves only on the
trainer's approval.

## Data model

### ChromaDB collections

- `docs` (configurable via `CHROMA_COLLECTION`)
  - **id**: string, e.g. `notes-0`, `notes-1`, …
  - **document**: string, the chunk text
  - **metadata**: `{ "source": "notes" }`

- `faq` (configurable via `CHROMA_FAQ_COLLECTION`)
  - **id**: string of the `faqs.id` primary key
  - **document**: the question
  - **metadata**: `{ "answer": "...", "faq_id": 42 }`

Both stored on disk in `./.chroma/`. Delete the folder to start over.

### External app integrations

The `/query` endpoint accepts an optional `apps` flag (`{weather, wiki}`) and
`location` (`{lat, lon}`). When enabled and the question matches a keyword
pattern (e.g. `weather in <city>`, `who is …`, `what is …`), the backend falls
through to:

- **Weather** — Open-Meteo geocoding + forecast (no API key). If the client
  provides `lat`/`lon`, the geocoding step is skipped and the forecast endpoint
  is called directly.
- **Wikipedia** — REST `page/summary` endpoint (no API key), with an
  opensearch fallback for ambiguous titles.

The response carries a new `app` payload and a `source` of `weather` or `wiki`
so the UI can render source tags and links.

### Postgres tables

`documents`:

| Column        | Type          | Notes                       |
| ------------- | ------------- | --------------------------- |
| `id`          | SERIAL        | Primary key                 |
| `source`      | VARCHAR       | Logical source name         |
| `chunk_index` | INTEGER       | Position within that source |
| `text`        | VARCHAR       | Chunk text                  |
| `created_at`  | TIMESTAMP     | Default `now()`             |

`faqs` (trained Q↔A, written by the trainer or by the inline ✏ correction):

| Column        | Type          | Notes                                  |
| ------------- | ------------- | -------------------------------------- |
| `id`          | SERIAL        | Primary key, also used as Chroma id    |
| `question`    | TEXT          | The canonical question                 |
| `answer`      | TEXT          | The trained answer                     |
| `created_at`  | TIMESTAMP     | Default `now()`                        |
| `updated_at`  | TIMESTAMP     | Updated on edit, mirrored to Chroma    |

`pending_questions` (questions the bot could not answer, awaiting training):

| Column        | Type          | Notes                                  |
| ------------- | ------------- | -------------------------------------- |
| `id`          | SERIAL        | Primary key                            |
| `question`    | TEXT          | The unresolved question                |
| `occurrences` | INTEGER       | How many times it was asked            |
| `resolved`    | BOOLEAN       | True once a trainer provides an answer |
| `created_at`  | TIMESTAMP     | First seen                             |
| `last_seen_at`| TIMESTAMP     | Most recent occurrence                 |

The first table is metadata for document chunks; the other two back the training flow. The actual search is in Chroma.

## Configuration (`app/config.py`)

All knobs live in `.env`:

| Var                 | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `DATABASE_URL`      | SQLAlchemy DSN for Postgres                      |
| `CHROMA_PERSIST_DIR`| Where Chroma writes its files                    |
| `CHROMA_COLLECTION` | Collection name                                  |
| `SPACY_MODEL`       | spaCy model to load (sm / md / lg)               |
| `CORS_ORIGINS`      | Comma-separated list of allowed frontend origins |
| `BASE_ANSWER_CONFIDENCE` | The confidence a fresh answer starts with  |
| `MAX_ANSWER_CONFIDENCE`  | The highest confidence an answer can reach |
| `CONFIDENCE_STEP`   | How much a correction raises the confidence      |

`settings.cors_list` splits the CSV into a list for the CORS middleware.

## Extending the system

Common next steps, in order of complexity:

1. **Train with files (PDF, DOCX, MD).** Add an `/ingest-file` endpoint that reads the upload, extracts text, then reuses `ingest_text`.
2. **Better embeddings.** Swap Chroma's default embedding function for a sentence-transformers model.
3. **Add an LLM.** Take the top-k chunks, build a prompt, call OpenAI / Ollama, return the answer instead of raw chunks.
4. **Multi-user.** Add a `users` table and a `user_id` FK on `documents`; pass a `X-User-Id` header.
5. **Source management.** Add `GET /sources` and `DELETE /sources/{name}` to inspect and reset training data.
6. **Streaming answers.** Use FastAPI's `StreamingResponse` and an LLM streaming endpoint.

## Conventions

- **Code style:** PEP 8, 4-space indent, type hints on public functions.
- **Imports:** direct submodule imports only (`from app.services.nlp import chunk_text`). The `__init__.py` files are intentionally empty to avoid circular imports.
- **No comments** unless something is non-obvious; this is enforced by the project's code-style rule.
- **No commits** to git unless explicitly requested.