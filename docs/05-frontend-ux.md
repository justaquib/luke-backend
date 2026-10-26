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
`Assistant`). The bar shows the name, scope and the page's live context title
(a chat's session title, renamed in place); the info panel shows the title and
description. A page with secondary controls registers them through a
`<ng-template appPageInfoActions>` that the info panel draws below the page's
facts, and its one primary action through `<ng-template appTopBarActions>`,
which the bar draws before the frame controls. A page's live facts register
through `<ng-template appPageInfoFacts>` and sit at the top of the panel, so
the page body keeps only its content. The panel shows the info, the facts, the
controls and the engine facts; the details control switches to the details the
current page registered and the info control switches back. A running
background task is pinned below the active content. The panel starts below the
top panel and
closes with the backdrop, the close button or Escape. A running background task
also shows at the bottom of the open primary sidebar. A page never repeats its
title, eyebrow, description or actions row in its body.

## Chat page (`/chat`) — the main UX

```
┌──────────────┬─────────────────────────────────────────────┐
│ ☰ Luke       │ Chat · ASSISTANT · Why is the sky blue  [i][⚙][▤] │  top panel
├──────────────┼─────────────────────────────────────────────┤
│  + New chat  │  Assistant: Hello! I am your Luke assistant…│
│              │                                             │
│  • Chat #1   │  You: Why is the sky blue?                  │
│  • Chat #2   │  Assistant: The sky appears blue because…   │
│              │  Source: Documents     [i] [copy] [👍] [👎] │
│              │                                             │
│              ├─────────────────────────────────────────────┤
│              │ [Type a message…]                     [➤]   │
└──────────────┴─────────────────────────────────────────────┘
```

### Components in the chat page

| Area          | What it does                                                                    |
| ------------- | ------------------------------------------------------------------------------- |
| Top panel     | Shared frame: brand + navigation, page name + scope + session title, info/settings/details |
| Sidebar       | Chat sessions, "New chat", the running background task at the bottom             |
| Messages      | Flat answers; an answer opens its details in the right panel; provenance and copy / good / wrong actions |
| Composer      | Model picker (Luke or an installed teacher) above the textarea + send button    |
| Right sidebar | Page information (the conversation facts for the active session, then the engine facts) or answer details (question, answer, source, measured distance and confidence, model, match, passages, "How this answer was found", the selected model's attention heatmap and next words, and the live link), with running tasks pinned below |

The conversation facts — message count, distinct answer sources and start time
— live in the page information panel, not the body. They are read from the
active session, so the panel shows the session the top bar names; switching
sessions (with the panel closed, since the panel is modal) and reopening it
shows that session's measured values.

### The live answer trace

An answer is streamed from `POST /query/stream`, and the page draws only the
stages the backend actually produced:

1. **Reading** — each word of the question with its lemma, part of speech,
   meaning and synonyms from the offline dictionary.
2. **Thinking** — the question type, the key words and the plan of sources.
3. **Connecting dots** — one row per source as it is consulted, with the real
   outcome: a match with its distance, evidence or matched question, a miss, or
   that the source is not enabled.
4. **The answer** — shown word by word once it has fully arrived. The reveal
   presents an answer that is already complete; it never invents a stage or a
   word. Under reduced motion the answer appears at once, and the full text is
   in the DOM for screen readers while the visible part grows.

The trace is stored with the message and shown in the answer details under
**How this answer was found**. A stored trace is shown in full when the panel
opens; it does not replay the word-by-word reveal. A failed stream clears the
live trace and adds no answer.

### The model picker

The composer asks one model to answer: **Luke** (the taught material) or any
Hugging Face teacher saved on this machine (`GET /language-model/models`). The
choice is kept in this browser (`luke.answer-model`); a saved teacher that was
removed falls back to the backend's default, so the picker is never empty. The
chosen id is sent with `/query` and `/query/stream`, and the answer stores the
model and its display name. A teacher answers from the same retrieved lesson or
FAQ as Luke — it writes the wording from that reference, it does not invent
material — and its stage appears as **Writing** with the model's name.

### Distance, confidence and the model

The provenance line under an answer reads `Source <label>` and, only when the
backend measured them, `dist <0.10>` (the vector distance of the matched
answer), `confidence <100%>` and `model <name>`. Luke's lesson answers keep the
confidence they stored; a distance is shown only where a real vector distance
exists. Nothing is estimated in the page.

### The attention heatmap and the prediction lab

The answer details end with what the selected model measured for the question:

- `app-word-heatmap` draws one cell per word pair from the model's attention
  (last layer, averaged over heads, 12-token window). Rows are the word
  looking, columns the word it looked at; the colour is scaled to the strongest
  cell of this prefix, and the caption says so. A model that returns no
  attention shows a plain "did not report attention" line instead of an empty
  grid.
- **Open the prediction lab** opens a modal where the model continues one word
  at a time. The words and probabilities are the backend's top five; the
  animation only staggers the reveal of data already returned. The trainer can
  edit the prefix, pick a suggested word, remove a picked word (and continue
  from there) or type a replacement; every choice asks `POST
  /language-model/predict` again from the new prefix. Under reduced motion the
  chips appear without the stagger.

### Sessions (chats)

- Each session has a `title`, list of `messages`, and is identified by a short `id`.
- All sessions are persisted to `localStorage` under `assistance.sessions.v1` and `assistance.active.v1`. They survive page reloads.
- New sessions are titled "New chat" until the first user question, then the title becomes the first 40 chars of that question.
- Deleting a session prompts for confirmation.

## The parenting coaching chat (`/dashboard/parenting`)

The coaching chat is the place where the coach that talks the parents lives:

- The parent sends a turn (`POST /parenting/chat/session/{id}/turn/stream`) and
  the coach answers from the material the tenant taught, with the words, the
  grammar and the sentences of the lessons instead of a rule.
- The turn streams the same live trace as the chat: Reading, Thinking and
  Connecting dots, then the coach answer word by word. The trace carries the
  understanding the backend measured; nothing is invented by the page.
- A session is open while the conversation runs and answered once the coach
  has answered; the coach status chip says which. There is no correction,
  scaffold or "Needs a better answer" block: the coach serves taught answers,
  and a parent who wants a better answer teaches it in the Training room.
- A failed stream clears the live trace and leaves the session open, so the
  parent can send the turn again.

The page is `/dashboard/parenting`, it reuses the frame (top panel, right
sidebar, scroll, composer) and adds the coach bubble and the coach status.

## The learner page (`/dashboard/learner`)

The learner page teaches and quizzes the learner, and its **Try a question
with the learner** card is a preview of the same answer path the chat uses.
The preview streams from `POST /language-model/preview/stream`: the live trace
shows Reading, Thinking and Connecting dots, then the answer is revealed word
by word. When the learner's material matches more than one Trainer question,
the preview keeps its match prompt so the trainer can choose the intended
answer; the chosen answer keeps the trace above the existing **How I thought
about it** reasoning block. Trainer-approved answers are checked first, then
the learner's own starter material, words, sentences and lessons — uploaded
documents and web sources are not consulted.

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
2. The **Inbox** tab lists only the pending questions the bot couldn't answer, sorted by how often it was asked. Answered questions live under **Trained answers**, so the inbox has no resolved filter and no pending/total counter.
3. Type the answer, click **Save & train** — the same flow as above.

The page's info panel (the `i` control in the top bar) carries the corpus facts — trained answers, pending questions and their sum — read from the loaded lists, so the counts stay off the tab body. It holds no upload or refresh controls: uploading happens from the tile in the Documents tab, and the corpus refreshes after each action.

### 3. Add a Q↔A from scratch

In `/dashboard/trainer`, the **+ New** tab. Use it when you know the Q↔A ahead of time.

### 4. Train the word model

The brain page (`/dashboard/brain`) trains the word model from every source
above plus the lessons, documents, words and sentences. It is what the
learner's sentence feature uses. The Sources panel says what it will train
from before the run, and the training happens in its own process so the rest
of the app stays usable.

### The level ladder

The brain page shows the whole road from `0.0.0` to `1.0.0` as a game ladder:
101 rungs in ten bands, from Foundations to Judgment, ending at the Generalist.
A rung is marked **passed** when its level was approved, **ahead** when it was
measured and the approval is waiting, **next** only when it is the rung after
the approved level and a test for it exists, and **locked** otherwise. A locked
rung says that no test has been written for it yet and never shows a
measurement. The displayed level still moves only on the trainer's approval,
and measurement stops at `0.1.0` until the higher rungs have real tests.

### 5. Upload source documents

The **Documents** tab in `/dashboard/trainer` lists every file the assistant
answers from. Its first card is the upload tile: click it, or tab to its file
input and press Enter, to pick a csv, xlsx, docx, doc, pdf or txt file. The
tile shows the indexing state while `POST /upload` runs, and the refreshed file
cards follow it with their chunk counts and view, download and remove actions.

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
- **Streaming answers** — stream through the existing `AnswerStreamService` and render the shared `app-answer-pipeline` and `app-answer-reveal`; a stage is drawn only when the backend sent it.