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
   - Cleans whitespace.
   - Loads `en_core_web_sm` (cached after first use).
   - Adds the `sentencizer` pipe if no parser is present.
   - Splits into sentences and packs them into chunks (max 800 chars, 100-char overlap).
3. Each chunk is added to ChromaDB with a stable id like `notes-3` and a `{source}` metadata.
4. The same chunks are inserted as rows in the Postgres `documents` table for audit/search.
5. Response: `{source, chunks, start_id}`.

### Query (`POST /query`)

1. Client sends `{question, n_results}`.
2. `vector_store.search_faq` looks for a trained answer in the `faq` collection. If the best match has a cosine distance ≤ `FAQ_MATCH_THRESHOLD` (default `0.55`), the trained answer is returned with `source: "faq"`.
3. Otherwise `vector_store.query` falls back to the `docs` collection, returning the top-k closest chunks.
4. If the docs collection is also empty, the question is logged to `pending_questions` (occurrence counter incremented if it was already there) and a "trainer has been notified" placeholder is returned with `source: "pending"`.
5. `nlp.extract_entities` runs spaCy NER over the question to return any detected entities (e.g. `GPE` for countries) for the frontend to display.
6. Response: `{question, source, answer, hits, entities, pending_id?}`.

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
| `POST /language-model/brain/correction` | Teach the right answer to a question answered wrongly    |
| `POST /language-model/brain/train`     | Train the brain on the tenant's own material               |
| `POST /language-model/brain/test`      | Run the level tests again and report the rungs it passed   |
| `POST /language-model/lessons/check`   | Report what the learner already knows about a lesson       |
| `GET /language-model/lessons/duplicates` | The lessons that repeat each other, grouped for review    |
| `POST /language-model/lessons/duplicates/accept` | Keep every lesson of a group: the repeat is on purpose |
| `POST /language-model/lessons`         | Teach a lesson, or a story, a lesson or a topic            |
| `PUT /language-model/lessons/{id}`     | Edit a lesson the learner already holds                    |
| `DELETE /language-model/lessons/{id}`  | Remove a lesson                                            |
| `POST /language-model/preview`         | Answer a question with the learner's own material only     |
| `POST /language-model/test/start`      | Start a mock test, a final exam, or a trainer led test     |
| `POST /language-model/test/question`   | Ask the learner a question during a grade test             |
| `POST /language-model/test/score`      | Score the learner's last answer                            |
| `POST /language-model/test/finish`     | Close the test and open the next grade when it is passed   |
| `GET /language-model/dictionary`       | Look a word up in the offline dictionary                    |
| `POST /language-model/dictionary/learn`| Teach the learner a word and the meaning it should keep    |
| `POST /language-model/sentence`        | Build a sentence from the words the learner knows          |

Every route needs the same bearer token as the rest of the API and only ever
reads or writes the tenant of the signed in user.

A session lasts a week. `POST /auth/refresh` gives a signed-in user a new
token, so the app renews the session while it is in use instead of asking the
user to sign in again every day.

A lesson is a question and an answer. A story, a lesson or a topic is taught
as one passage instead: it is filed under the name it was given, every
sentence of it is taught on its own as well, and the learner can ask for it
by that name or by its topic.

Training the brain takes about a minute, so it runs in its own process: the
app stays usable while the model learns. The tenant's learner file carries a
`training-<tenant>.json` marker while that runs, and the page asks the brain
again until the marker is gone. The transformer is written next to the model
and the page picks it up as soon as the file changes, so the vocabulary grows
with every lesson that was taught.

### What the brain trains from

The training is built from every source the trainer has given, and they are
collected in one place, `app/services/language/training_materials.py`:

| Source | What it is |
| --- | --- |
| Lessons | The question and the answer of every lesson in the learner, and the starter lessons of the grades it reached. |
| Approved answers | The questions and the answers the trainer approved in the Trainer. |
| Corrections | The answers the trainer corrected, so the model stops making the same mistake. |
| Documents | The chunks of every uploaded document. |
| Words | The words and the meanings that were taught with the dictionary. |
| Sentences | The sentences that were taught on their own. |

Every source becomes two things: a sentence, so the words of it reach the
vocabulary, and a request with its answer, so the model learns to answer a
request it was never worded for. The questions and the answers of the approved
answers are learned this way as well, so the trainer's own answers are not
only looked up but written in the model's own words.

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

### Answering a request, not a question

A lesson is taught as a question and an answer, so a lookup finds it. A
request is different: "write a short poem about the rain" shares almost no
words with the lesson "Poem: Rain Rain Go Away", so no amount of question
matching finds it.

The brain answers requests in three steps.

| Step | Where | What happens |
| --- | --- | --- |
| Read the request | `app/services/language/shapes.py`, `requests.py` | The request is matched against the shapes it knows: a poem about a topic, a list of a topic, some facts about a topic. The topic is taken out of it. |
| Find the material | `app/services/language/topics.py`, `requests.py` | The lesson about that topic is found through the topic index. A word that many lessons share does not name any one of them, so it is not in the index: the assistant refuses rather than choosing one at random. |
| Write the answer | `app/services/transformer/` | The trained model writes the answer from the lesson it was given. |

The transformer is trained on both kinds of example, and the loss is taken
over the answer alone:

- a plain sentence, so the model keeps writing in the trainer's style,
- a request and its answer, so the model answers what it is asked,
- a request, the material it may use, and the answer.

Taking the loss over the answer alone is what makes the model answer a
request: the request and the material are read, never scored. The words of a
request are marked with `<instr>`, the material with `<ctx>`, and the answer
with `<resp>`, so the model knows which part is which.

The training is measured against the material, not a fixed number: its steps
grow so that the model makes several passes over the examples, because the
measured runs showed a tiny model learns to guess in one pass and only starts
knowing its lessons in the third and fourth. The loss on the brain page is
read in the model's resting state over a few fixed batches, so it says how
well the model knows the material and not how one draw of the training went.

Every lesson is asked for in many wordings, so the model learns the subject
rather than one phrasing of it. A subject belongs to one lesson only: when two
lessons claim the same subject, the one that asks about it the most directly
keeps it, so the model is never taught two different answers to one request.

Every word of a written answer is checked against the material it was given.
An answer that uses a word the trainer never taught is refused, and the taught
answer is used instead. A request about something never taught is refused as
well: the brain says it does not know instead of inventing.

### The system administrator in the chat

The system administrator has no tenant of their own: they look after every
tenant instead of belonging to one. The chat answers them anyway, reading the
material of the tenant-less bucket (where the lessons they taught themselves
live) and, with no tenant filter on it, the approved answers and the
documents of every tenant they look after. A user without a tenant is still
turned away, because they have no assistant of their own.

### Correcting an answer

`POST /language-model/brain/correction` takes the question, the answer that
was given and the answer that should have been given. It:

- stores the right answer as a trained answer of the tenant, so the next
  question of that kind is answered correctly from the FAQ collection,
- keeps the correction in the brain's own memory, so the writing model learns
  the right wording and the heads learn from it,
- asks the question again and checks that the assistant keeps the correction,
  so the trainer is told whether it really learned it.

### The instruction test

The brain page shows how well the model answers requests it was never
worded for. The requests used for the measure are held out of the training,
so a model that only memorises scores nothing on them. The score is in
`model_level.instruction_following`, and the requests themselves are in
`model_level.instruction_probes`.

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