# Frontend Connection Guide — Angular (overview)

> For a full tour of the UI (sidebar, train mode, settings, sessions), read `docs/05-frontend-ux.md`.

Step-by-step to wire your Angular app (`assistance_front/`) to the FastAPI backend in `assistance_back/`.

## 1. Confirm the backend is reachable

From the same machine:

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}
```

The backend already allows CORS from `http://localhost:4200` (the default Angular dev server). If your Angular port is different, edit `CORS_ORIGINS` in `assistance_backend/.env` and restart uvicorn.

## 2. What's already wired

The Angular project already includes the API integration:

- `src/app/core/api.service.ts` — typed wrapper around the base endpoints (`/health`, `/query`, `/ingest`, `/stats`, `/faq`, `/faq/inbox`, etc); the language-model calls are in `language-model-api.service.ts`.
- `src/environments/environment.ts` — `apiUrl: 'http://localhost:8000'` (dev).
- `src/environments/environment.production.ts` — empty `apiUrl` (use a reverse proxy).
- `src/app/shared/` — the one frame: `app-shell`, `app-sidebar`, `app-top-bar` (brand + navigation, page name + scope, info/settings/details), the content-aware `app-right-sidebar` with its pinned background tasks, and the shared facts and actions bands.
- `src/app/chat-ui/` — the assistant surface at `/chat` with sessions, train mode and inline corrections.
- `src/app/trainer/`, `learner/`, `brain/`, `expressions/`, `settings/` and `design/` — the builder pages under `/dashboard`, all inside the shared frame.
- `src/app/settings/` — Lab settings: model identity (`GET`/`PUT /language-model/settings`, the model's name and starting version), the foundation checkpoint (`GET /language-model/foundation`), engine connection, live sources, appearance and the System Overview package list.
- `src/app/expressions/` — the conversation expressions page at `/dashboard/expressions` (`GET`/`POST /language-model/expressions`, `PUT`/`DELETE /language-model/expressions/{id}`).
- `proxy.conf.json` — dev server proxies `/api/*` to `localhost:8000` (wired into `angular.json`).

If you start from a clean checkout, none of this is in the way — re-use or replace as needed.

## 3. Production deployment

Two common setups:

**Option A — same-origin reverse proxy (recommended for SSR):**

```
Angular SSR (port 4000)  ──▶  Nginx / Caddy
                                  │  /api/*  ──▶  FastAPI :8000
                                  │  /*      ──▶  Angular static / SSR
```

`environment.production.ts` keeps `apiUrl: ''`, so calls become relative `/api/query`. Add a small interceptor that prefixes `/api`, or update the service:

```ts
private readonly base = environment.production ? '/api' : environment.apiUrl;
```

**Option B — separate origins:**

- Update `CORS_ORIGINS` in the backend `.env` to your deployed Angular origin.
- Set `apiUrl` in `environment.production.ts` to the deployed API URL.

## 4. Production checklist

- [ ] Backend served behind HTTPS at a real domain.
- [ ] Update `CORS_ORIGINS` in the backend `.env` to your deployed Angular origin.
- [ ] Lock down Postgres credentials; don't keep `postgres/postgres` in production.
- [ ] Run uvicorn behind a production server (e.g. `gunicorn -k uvicorn.workers.UvicornWorker`) or use Docker.
- [ ] Back up the `./.chroma` folder (vector store) and the Postgres `documents`, `faqs`, and `pending_questions` tables.