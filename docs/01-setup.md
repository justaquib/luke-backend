# Project Setup Guide — Python Backend

A complete step-by-step to get the Python/FastAPI backend running on a fresh machine.

## Prerequisites

- **Python 3.10+** (we used 3.12)
- **pip** and **venv**
- **Docker** + **Docker Compose** (for PostgreSQL)
- **curl** (to test the API)
- ~500 MB free disk space (for the venv + spaCy model)

Check your versions:

```bash
python3 --version
docker --version
docker compose version
```

## 1. Create / enter the project folder

```bash
cd Documents/assistance
```

The project files are already in place. If you started from a clean clone, copy them in first.

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your shell prompt. On Windows use `.venv\Scripts\activate`.

## 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- `fastapi` + `uvicorn` — web framework + ASGI server
- `pydantic` + `pydantic-settings` — validation and env-based config
- `sqlalchemy` + `psycopg2-binary` — Postgres ORM + driver
- `chromadb` — local vector database
- `spacy` — NLP pipeline

## 4. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

The `_sm` model is small/fast. For better accuracy later you can switch to `_md` or `_lg` in `.env` (`SPACY_MODEL=en_core_web_md`).

## 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and adjust if needed:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5433/assistance
CHROMA_PERSIST_DIR=./.chroma
CHROMA_COLLECTION=docs
SPACY_MODEL=en_core_web_sm
CORS_ORIGINS=http://localhost:4200
```

> Port **5433** is used because the docker container maps to host 5433. If you change it, update both the `docker-compose.yml` `ports` mapping and `DATABASE_URL` to match.

## 6. Start PostgreSQL (Docker)

```bash
docker compose up -d
```

Verify it is up:

```bash
docker ps --filter name=assistance-pg
```

You should see `assistance-pg` with status `Up` and ports `0.0.0.0:5433->5432/tcp`.

First-time only — the `assistance` database is created automatically by SQLAlchemy on app startup. If it complains about the database not existing, create it manually:

```bash
docker exec -it assistance-pg psql -U postgres -c "CREATE DATABASE assistance;"
```

## 7. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

## 8. Smoke-test the endpoints

In another terminal:

```bash
# Health
curl http://localhost:8000/health

# Ingest some text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source":"notes","text":"FastAPI is a modern Python web framework. ChromaDB stores embeddings locally."}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI?", "n_results":3}'

# Stats
curl http://localhost:8000/stats
```

Or open the auto-generated interactive docs at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 9. Stop the project

```bash
# Stop the API: Ctrl+C in the uvicorn terminal
# Stop the database
docker compose down
# (add -v to also delete the database volume)
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `password authentication failed for user "postgres"` on `5432` | System Postgres is on 5432; use our Docker on 5433 (or change `docker-compose.yml` and `DATABASE_URL`). |
| `ModuleNotFoundError: No module named 'app.services.nlp'` | You skipped step 3 or the venv is not active. |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | Run step 4. |
| Port 8000 already in use | Run uvicorn with `--port 8001` and update your frontend. |
| `Failed to send telemetry event … capture() takes 1 positional argument but 3 were given` | Posthog / Chroma version mismatch. Telemetry is disabled by the `.env` values `ANONYMIZED_TELEMETRY=False` and `POSTHOG_DISABLED=True`; restart uvicorn so they take effect. |