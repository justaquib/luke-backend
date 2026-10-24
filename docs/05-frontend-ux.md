# Frontend Guide — App Layout & Training UX

Walks through the Angular app shell, the chat UX, and every way a user (or trainer) can teach the assistant.

## Project layout

```
assistance_front/
├── src/
│   ├── app/
│   │   ├── app.ts / .html / .css      Root shell (just <router-outlet />)
│   │   ├── app.config.ts              provideHttpClient + provideRouter + SSR
│   │   ├── app.routes.ts              public entry pages, /chat, guarded /dashboard/*
│   │   ├── core/                      API, session, design, appearance and preference services
│   │   ├── shared/                    The frame: app-shell, app-sidebar, app-top-bar,
│   │   │                              app-right-sidebar, page info and actions, app-engine-facts,
│   │   │                              app-modal, toasts, icons and background tasks
│   │   ├── chat-ui/                   Assistant surface (/chat)
│   │   ├── brain/                     Assistance brain (/dashboard/brain)
│   │   ├── learner/                   Language learner (/dashboard/learner)
│   │   ├── trainer/                   Training room (/dashboard/trainer)
│   │   ├── parenting/                 Parenting coach (/dashboard/parenting)
│   │   ├── settings/                  Model identity, engine, sources and appearance
│   │   ├── design/                    Workspace design control room
│   │   └── tenants/                   Tenant administration
│   └── environments/
│       ├── environment.ts             dev: apiUrl http://localhost:8000
│       └── environment.production.ts  empty apiUrl (reverse-proxy expected)
├── proxy.conf.json                    /api/* → http://localhost:8000
└── angular.json                       references both
```

## The frame — top panel, info sidebar and right sidebar

`app-shell` and the chat frame draw the shared top panel (`app-top-bar`) on
every page in every sidebar state. It has three zones:

- **left** — the brand (logo and assistant name) and the primary-sidebar
  toggle;
- **middle** — the page name, a small accent dot and the page's scope,
  centered (`Assistance brain` · `Model builder`, `Settings` · `Engine
  setting`);
- **right** — the page's one primary action when it has one (the brain's
  icon-only play button posts a training and opens its sources), then info
  (opens the page info and engine facts), settings (opens
  `/dashboard/settings`) and details (opens the shared right sidebar on the
  page's registered details).

Page identity lives in the top bar and the info sidebar, not in the page body.
`pageInfoForUrl` maps each route to a name, scope, title and description
(`Assistance brain` / `Neural engine` / `Model builder`, `Chat` / `Chat` /
`Assistant`). The bar shows the name and scope; the info panel shows the title
and description. A page with secondary controls registers them through a
`<ng-template appPageInfoActions>` that the info panel draws at its top, and
its one primary action through `<ng-template appTopBarActions>`, which the bar
draws before the frame controls. The panel shows the info, the controls and
the engine facts; the details control switches to the details the current page
registered and the info control switches back. A running background task is
pinned below the active content. The panel starts below the top panel and
closes with the backdrop, the close button or Escape. A running background task
also shows at the bottom of the open primary sidebar. A page never repeats its
title, eyebrow, description or actions row in its body.

## Chat page (`/chat`) — the main UX

```
┌──────────────┬─────────────────────────────────────────────┐
│ ☰ Luke       │ Chat · ASSISTANT                 [i] [⚙] [▤] │  top panel
├──────────────┼─────────────────────────────────────────────┤
│  + New chat  │ Conversation engine       ● Engine online   │
│              │ ┌ Messages · Answer paths · Started ──────┐ │
│  • Chat #1   │                                             │
│  • Chat #2   │  You: What is ChromaDB?                     │
│              │  Assistant: ChromaDB is a vector database…  │
│              │  Source: ...          [copy] [👍] [👎]      │
│              │                                             │
│              ├─────────────────────────────────────────────┤
│              │ [Type a message…]                     [➤]   │
└──────────────┴─────────────────────────────────────────────┘
```

### Components in the chat page

| Area            | What it does                                                        |
| --------------- | ------------------------------------------------------------------- |
| Top panel       | Shared frame: brand + navigation, page name + scope, info/settings/details |
| Sidebar         | Chat sessions, "New chat", the running background task at the bottom |
| Chat header     | Session title, engine state and the settings shortcut               |
| Facts strip     | Messages, answer paths and the start time of the conversation       |
| Messages        | Flat answers with provenance and copy / good / wrong actions         |
| Composer        | Textarea + send button                                              |
| Right sidebar   | Page details or engine information, with running tasks pinned below |

### Sessions (chats)

- Each session has a `title`, list of `messages`, and is identified by a short `id`.
- All sessions are persisted to `localStorage` under `assistance.sessions.v1` and `assistance.active.v1`. They survive page reloads.
- New sessions are titled "New chat" until the first user question, then the title becomes the first 40 chars of that question.
- Deleting a session prompts for confirmation.

## The parenting coaching chat (`/dashboard/parenting`)

The coaching chat is the place where the coach that talks the parents lives:

- The parent sends a turn (`POST /parenting/chat/session/{id}/turn`) and the
  coach answers from the material the tenant taught, through the same writer
  the answering model uses, so the words, the grammar and the sentences come
  from the lessons instead of a rule.
- The answer is read with the writer's own eyes: its confidence, the words it
  was unsure about, the grammar and the loop. When the answer is not good
  enough, the turn ends in a scaffold (`status: needs_correction`) with the
  two coaching phrasings the parent can pick from, and the coach retries with
  the chosen correction (`POST /parenting/chat/session/{id}/correction`).
- After the retry passes the gate the answer is kept for the brain, so the
  coaching is itself training material.
- Repeated scaffolds escalate the session (`status: escalated`) and hand the
  parent to a human trainer.

The page is `/dashboard/parenting`, it reuses the frame (top panel, right
sidebar, scroll, composer) and adds the coach bubble, the scaffold panel and
the coach status.

## Training — ways to teach the assistant

### 1. Correct a wrong answer from the chat

This is the primary path:

1. Find the assistant answer that was wrong and click the thumbs-down action.
2. The app opens the Training room (`/dashboard/trainer`) with the question
   and the wrong answer already in the teaching form.
3. Write the right answer and save it. The Q↔A goes to `POST /faq`, so the
   next time anyone asks that question (or one close enough — see
   `FAQ_MATCH_THRESHOLD`) the trained answer is returned.

### 2. Review the inbox (bulk)

Best for catching patterns you missed:

1. Go to `/dashboard/trainer`.
2. The **Inbox** tab shows every pending question the bot couldn't answer, sorted by how often it was asked.
3. Type the answer, click **Save & train** — the same flow as above.

### 3. Add a Q↔A from scratch

In `/dashboard/trainer`, the **+ New** tab. Use it when you know the Q↔A ahead of time.

### 4. Train the writing model

The brain page (`/dashboard/brain`) trains the neural writer from every source
above plus the lessons, documents, words and sentences. The Sources panel says
what it will train from before the run, and the training happens in its own
process so the rest of the app stays usable.

## Settings page

Settings is `/dashboard/settings`, reached from the settings control in the top
panel. It has four sections, in this order:

1. **Model identity** — the model's name (default `Luke`) and the version it
   starts from (default `0.0.0`). The name is used wherever the assistant
   speaks about itself, including the greeting, and the starting version is
   the floor of the level ladder. Saved through
   `GET`/`PUT /language-model/settings`.
2. **Engine connection** — the backend URL and the live connection state.
3. **Live sources** — what the chat can answer from and where it falls back.
4. **Appearance** — colour mode, text size, typeface and palette.

Each card has one info button; the explanations do not take up the page.

## Background tasks in the frame

A running background task is shown in two places: at the bottom of the open
primary sidebar, and pinned at the bottom of the open right sidebar. The app
root starts `BackgroundTasksService` once while signed in; it polls fast while
a task runs and slowly while idle, so a task that starts later still appears.
The right sidebar shows the task section only while a task runs; it never
replaces the page details or the engine information.

## State summary

| State                | Stored where                                | Survives reload? |
| -------------------- | ------------------------------------------- | ---------------- |
| Chat sessions        | `localStorage` (`assistance.sessions.v1`)   | Yes              |
| Active session id    | `localStorage` (`assistance.active.v1`)     | Yes              |
| Frame design         | `localStorage` (`assistance.design.v1`)     | Yes              |
| Appearance           | `localStorage` (`assistance.appearance.v1`) | Yes              |
| Model name / version | Backend `model-<tenant>.json`               | Yes, per tenant  |
| Sidebar open state   | In-memory `signal` in `DesignService`       | No               |
| Backend status       | Re-read by the chat service                 | n/a              |
| Trained FAQ entries  | Postgres `faqs` + Chroma `faq` collection   | Yes              |
| Pending questions    | Postgres `pending_questions`                | Yes              |
| Document chunks      | Postgres `documents` + Chroma `docs`        | Yes              |

## Adding new features

- **Per-session backend override** — store `apiUrl` per session, swap the `ApiService` base via DI.
- **Export / import sessions** — read/write the `localStorage` keys.
- **A new builder page** — add it below the guarded `/dashboard` route, add its name and scope to `shared/page-title.ts`, and give it the frame rather than its own header.
- **Streaming answers** — switch to `HttpClient.stream` and use Server-Sent Events from FastAPI.