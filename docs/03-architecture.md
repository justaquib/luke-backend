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