# Frontend Connection Guide — Angular (overview)

> For a full tour of the UI (sidebar, train mode, settings, sessions), read `docs/05-frontend-ux.md`.

Step-by-step to wire your Angular app (`assistance_web/ag_frontend`) to the FastAPI backend in `assistance_backend/`.

## 1. Confirm the backend is reachable

From the same machine:

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}
```

The backend already allows CORS from `http://localhost:4200` (the default Angular dev server). If your Angular port is different, edit `CORS_ORIGINS` in `assistance_backend/.env` and restart uvicorn.

## 2. What's already wired

The Angular project already includes the API integration:

- `src/app/core/api.service.ts` — typed wrapper around all endpoints (`/health`, `/query`, `/ingest`, `/stats`, `/faq`, `/faq/inbox`, etc).
- `src/environments/environment.ts` — `apiUrl: 'http://localhost:8000'` (dev).
- `src/environments/environment.production.ts` — empty `apiUrl` (use a reverse proxy).
- `src/app/chat-ui/chat-ui.ts` — full chat shell with sidebar, sessions, train-mode toggle, inline ✏ corrections.
- `src/app/trainer/trainer.ts` — inbox / trained / +New tabs.
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