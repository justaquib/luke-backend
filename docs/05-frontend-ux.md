# Frontend Guide — App Layout & Training UX

Walks through the Angular app shell, the chat UX, and every way a user (or trainer) can teach the assistant.

## Project layout

```
assistance_web/ag_frontend/
├── src/
│   ├── app/
│   │   ├── app.ts / .html / .css      Root shell (just <router-outlet />)
│   │   ├── app.config.ts              provideHttpClient + provideRouter + SSR
│   │   ├── app.routes.ts              "/" → ChatUI, "/trainer" → Trainer
│   │   ├── core/
│   │   │   └── api.service.ts         Typed wrapper for all /api endpoints
│   │   ├── chat-ui/
│   │   │   ├── chat-ui.ts             Sidebar + chat area + train mode state
│   │   │   ├── chat-ui.html
│   │   │   └── chat-ui.css
│   │   └── trainer/
│   │       ├── trainer.ts             Inbox / Trained / +New tabs
│   │       ├── trainer.html
│   │       └── trainer.css
│   └── environments/
│       ├── environment.ts             dev: apiUrl http://localhost:8000
│       └── environment.production.ts  empty apiUrl (reverse-proxy expected)
├── proxy.conf.json                    /api/* → http://localhost:8000
└── angular.json                       references both
```

## Chat page (`/`) — the main UX

```
┌──────────────┬────────────────────────────────────────────┐
│  ☰  Assistance│  Title              [○ Train]  [⚙]      │
│              ├────────────────────────────────────────────┤
│  + New chat  │                                            │
│              │   Hello! I am your assistance…            │
│  • Chat #1   │                                            │
│  • Chat #2   │   ┌─ you ────────────────┐                │
│  • Chat #3   │   │ What is ChromaDB?    │                │
│              │   └──────────────────────┘                │
│  ● Connected │                                            │
│              │   ┌─ assistant ─────────┐ ✏              │
│              │   │ ChromaDB is a vector │                │
│              │   │ database…            │                │
│              │   └──────────────────────┘                │
│              │                                            │
│              ├────────────────────────────────────────────┤
│              │ [Type a message…]                  [➤]     │
└──────────────┴────────────────────────────────────────────┘
```

### Components in the chat page

| Area                | What it does                                                       |
| ------------------- | ------------------------------------------------------------------ |
| Sidebar             | List of chat sessions, "New chat" button, backend status, collapse |
| Chat header         | Session title, mode pill, **Train** toggle, **Settings** icon      |
| Messages            | Conversation; assistant bubbles can be corrected with ✏            |
| Composer            | Textarea + send button                                             |
| Settings panel      | Slides in under header (regular mode only)                          |
| Train banner        | Slides in under header in train mode                                |

### Sessions (chats)

- Each session has a `title`, list of `messages`, and is identified by a short `id`.
- All sessions are persisted to `localStorage` under `assistance.sessions.v1` and `assistance.active.v1`. They survive page reloads.
- New sessions are titled "New chat" until the first user question, then the title becomes the first 40 chars of that question.
- Deleting a session prompts for confirmation.

## Training — three ways to teach the assistant

### 1. Inline correction (the "the bot was wrong" path)

This is the primary path. It is meant to feel fast:

1. In the chat header, click **Train** to switch to Train mode (or click ✏ on any assistant message — it auto-toggles Train mode for you).
2. The whole UI switches to a **dark theme** so you can see at a glance you are in training.
3. A yellow banner appears: *"Train mode is on. Click ✏ on any assistant message to correct the answer."*
4. On the assistant bubble you want to fix, click ✏. The bubble turns into an inline editor pre-filled with the current answer.
5. Edit the answer, click **Save & train**.
6. The Q↔A is sent to `POST /faq` and the message is updated in place (with the **Trained** tag).
7. The next time anyone asks the same question (or one close enough — see `FAQ_MATCH_THRESHOLD`), the trained answer is returned.

### 2. Review the inbox (bulk)

Best for catching patterns you missed:

1. Go to `/trainer` (also dark themed).
2. The **Inbox** tab shows every pending question the bot couldn't answer, sorted by how often it was asked.
3. Type the answer, click **Save & train** — the same flow as above.

### 3. Add a Q↔A from scratch

In `/trainer`, the **+ New** tab. Use it when you know the Q↔A ahead of time.

## Settings panel

A simple slide-down panel under the header, shown when:

- You're not in Train mode
- You click the ⚙ icon

It shows the current backend URL and connection state. Wire it to a real `AppConfigService` when you need user preferences.

## Train mode behavior, in detail

- The whole `<div class="shell">` gets a `train-on` class which triggers the dark theme via CSS variables.
- The mode pill in the header changes from "Regular" to "Train mode".
- The Train button changes from `○ Train` to `◉ Train` and turns amber.
- All messages still scroll and the composer still works — you can ask new questions and immediately correct the new answers.
- Settings panel is hidden in Train mode (you can't accidentally misconfigure things while training).
- Switching Train off (click ◉ Train again) returns to the regular theme and reveals the Settings panel again.
- Train mode is per-page-load — it's not persisted (so a refresh leaves the regular theme).

## State summary

| State               | Stored where                              | Survives reload? |
| ------------------- | ----------------------------------------- | ---------------- |
| Chat sessions       | `localStorage` (`assistance.sessions.v1`) | Yes              |
| Active session id   | `localStorage` (`assistance.active.v1`)   | Yes              |
| Train mode (UI)     | In-memory `signal`                        | No               |
| Settings panel open | In-memory `signal`                        | No               |
| Sidebar collapsed   | In-memory `signal`                        | No               |
| Backend status      | Re-fetched on `ngOnInit`                  | n/a              |
| Trained FAQ entries | Postgres `faqs` + Chroma `faq` collection | Yes              |
| Pending questions   | Postgres `pending_questions`              | Yes              |
| Document chunks     | Postgres `documents` + Chroma `docs`      | Yes              |

## Adding new features

- **Per-session backend override** — store `apiUrl` per session, swap the `ApiService` base via DI.
- **Export / import sessions** — read/write the `localStorage` keys.
- **Keyboard shortcut for Train mode** — listen for `Ctrl+Shift+T` in the host listener.
- **Streaming answers** — switch to `HttpClient.stream` and use Server-Sent Events from FastAPI.