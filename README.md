# Luke Lab — self-trained RAG API

A small retrieval-augmented chatbot backend you can train on your own documents.

**Stack:** Python, FastAPI, ChromaDB, spaCy, PostgreSQL, SQLAlchemy.

## Layout

```
app/
  main.py            FastAPI routes (/health, /ingest, /query, /stats)
  config.py          Pydantic settings (reads .env)
  schemas.py         Request/response models
  db/database.py     SQLAlchemy engine + Document model
  services/
    nlp.py           spaCy sentence chunking + entity extraction
    vector_store.py  ChromaDB persistent client + query
    ingestion.py     Ingest text -> chunk -> embed -> store
samples/notes.txt    Example training text
docker-compose.yml   PostgreSQL 16
.env.example         Copy to .env
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env

# start postgres on port 5433 (avoids conflict with system Postgres on 5432)
docker compose up -d
```

## Run

From this folder:

```bash
./start engine
```

The word after the launcher is free-form: `./start engine`, `./start luke` or
any other name starts the same backend. The file can be renamed too, for
example `mv start launcher` and then `./launcher engine`.

The command uses `.venv` (creating it when it is missing), installs
`requirements.txt` only when the file changed, checks the spaCy model, starts
Postgres with Docker Compose (opening Docker Desktop when needed), waits for
the database, then runs uvicorn with reload on http://localhost:8000.

To type it from any folder, link it once in a folder on your `PATH`:

```bash
ln -s "$PWD/start" ~/.local/bin/start
start engine
```

After renaming the launcher, point the link at the new name:

```bash
ln -sf "$PWD/launcher" ~/.local/bin/start
```

The manual route still works:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Docs at http://localhost:8000/docs

## Training workflow

1. **Ingest** your own text (notes, docs, FAQs):

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source":"notes","text":"<paste your training text here>"}'
```

You can call `/ingest` many times with different `source` values to build up your knowledge base.

2. **Query** it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI?", "n_results":4}'
```

3. **Check size:**

```bash
curl http://localhost:8000/stats
```

## Connect your Angular app

The API already allows `http://localhost:4200` via CORS. In your Angular service:

```ts
private api = 'http://localhost:8000';

ask(question: string) {
  return this.http.post<any>(`${this.api}/query`, { question, n_results: 4 });
}

train(source: string, text: string) {
  return this.http.post<any>(`${this.api}/ingest`, { source, text });
}
```

If your Angular dev server runs on a different port, edit `CORS_ORIGINS` in `.env`.

## Documentation

Step-by-step guides in `docs/`:

- `docs/01-setup.md` — Python project setup from scratch
- `docs/02-frontend-connection.md` — wire up your Angular app
- `docs/03-architecture.md` — file layout, request flow, data model, extension ideas
- `docs/04-training.md` — FAQ learning flow, `/faq` API, trainer page, threshold tuning
- `docs/05-frontend-ux.md` — Angular shell layout, sessions, train-mode UX, settings
- `docs/06-frontier-model-evaluation.md` — the standard techniques the frontier labs measure with, and how this app maps to them
- `docs/07-model-measurements.md` — the app's own measured numbers, kept current after every improvement

## Notes

- Retrieval-only mode returns the most relevant chunks; wire your own LLM later if you want generated answers.
- spaCy runs locally; download `en_core_web_sm` (small) or upgrade to `en_core_web_md` / `en_core_web_lg` for better vectors later.
- ChromaDB persists to `./.chroma`; delete the folder to retrain from scratch.