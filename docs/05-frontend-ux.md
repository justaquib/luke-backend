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
│   │   ├── chat-ui/                   Luke chat surface (/chat)
│   │   ├── brain/                     Luke brain (/dashboard/brain)
│   │   ├── assessment/                Held-out assessment (/dashboard/assessment):
│   │   │                              run panel, eight skill cards and the rubric review
│   │   ├── learner/                   Language learner (/dashboard/learner)
│   │   ├── trainer/                   Training room (/dashboard/trainer)
│   │   ├── parenting/                 Parenting coach (/dashboard/parenting)
│   │   ├── settings/                  Lab settings: model identity, engine, sources,
│   │   │                              appearance and the System Overview
│   │   └── design/                    Lab design control room
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
  centered (`Luke brain` · `Model builder`, `Lab settings` · `Engine
  setting`);
- **right** — the page's one primary action when it has one (the brain's
  icon-only play button posts a training and opens its sources), then info
  (opens the page info and engine facts), settings (opens
  `/dashboard/settings`) and details (opens the shared right sidebar on the
  page's registered details).

Page identity lives in the top bar and the info sidebar, not in the page body.
`pageInfoForUrl` maps each route to a name, scope, title and description
(`Luke brain` / `Neural engine` / `Model builder`, `Chat` / `Chat` /
`Luke`). The bar shows the name, scope and the page's live context title
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
| Composer      | Rounded input panel: auto-growing textarea over a dashed divider, then a tools row with new chat and the live-sources menu on the left and the model picker and stop / send on the right |
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

The composer asks one model to answer: **Luke** (taught material first, then
the read-only foundation checkpoint when nothing is taught) or any Hugging Face
teacher saved on this machine (`GET /language-model/models`). The
choice is kept in this browser (`luke.answer-model`); a saved teacher that was
removed falls back to the backend's default, so the picker is never empty. The
chosen id is sent with `/query` and `/query/stream`, and the answer stores the
model and its display name. A teacher answers from the same retrieved lesson or
FAQ as Luke — it writes the wording from that reference, it does not invent
material — and its stage appears as **Writing** with the model's name.

### Taught, written and composed answers

Every answer says which kind it is, in the provenance line and in the details:

- **Taught** (`source` of a saved answer, lesson, expression or document) is
  served as the trainer wrote it. That is the normal answer.
- **Written** (`mode: "generated"`) has no taught match: the foundation
  checkpoint wrote it. The line says **Written from the base model, not from
  taught material. Unverified.** and the details show the model name and
  version. It carries no citation, and the page never presents it as taught.
- **Composed** (`mode: "hybrid"`) was written from numbered passages the
  reader supplied. The details list **Cited passages** with the index, source,
  label and excerpt, and the verifier's note. When the verifier rejects the
  wording, the passages themselves are served and the page says so.

While Luke writes, the stream shows **Writing** and the words arrive as
`token` events; **Stop** ends the stream and frees the model. The details panel
also carries the foundation card from Settings, so the reader can see which
checkpoint wrote the answer. The measured speed, memory and perplexity are in
`docs/07-model-measurements.md`.

### The composer panel

The input is one rounded panel, not a bare field: the textarea grows with the
draft up to its cap, a dashed divider separates it from the tools row, and the
panel takes the accent border while it has focus or an answer is streaming. On
the left the **+** button starts a new chat and **Sources** opens a menu of the
live sources (`Weather`, `Wiki`) that may answer when the taught material has
none; the pill names how many are on. On the right the model picker names the
model that answers, then **Stop** while an answer streams and the send button
otherwise. Send is disabled for an empty draft, so a second question cannot be
queued by accident; Enter sends and Shift + Enter starts a new line. The
Sources menu closes when the message box takes focus and on Escape.

### Distance, confidence and the model

The provenance line under an answer reads `Source <label>` and, only when the
backend measured them, `dist <0.10>` (the vector distance of the matched
answer), `confidence <100%>` and `model <name>`. Luke's lesson answers keep the
confidence they stored; a distance is shown only where a real vector distance
exists. Nothing is estimated in the page.

### The attention heatmaps and the prediction lab

The answer details end with what the selected model measured for **two** texts:
the **Question** that produced this answer and the **Answer** itself. Each read
shows its own attention heatmap and next-word list, so the trainer can compare
what the model expected while reading the question with what it expected while
writing the answer.

- `app-word-heatmap` draws one cell per word pair from the model's attention
  (last layer, averaged over heads, 12-token window). Rows are the word
  looking, columns the word it looked at; the colour is scaled to the strongest
  cell of this prefix, and the caption says so. A model that returns no
  attention shows a plain "did not report attention" line instead of an empty
  grid.
- **Open in lab** appears on each read and opens the prediction modal seeded
  with that text — the question or the answer. The tokens and probabilities are
  the backend's top five; the animation only staggers the reveal of data
  already returned. The trainer can edit the prefix, pick a suggested token,
  remove a picked token (and continue from there) or type a replacement; every
  choice asks `POST /language-model/predict` again from the new prefix. Under
  reduced motion the chips appear without the stagger. A BPE checkpoint is a
  piece model: the lab says **piece**, joins pieces without adding spaces, and
  never shows subword behaviour under the word label.
- A workspace whose word model has not been trained yet (the backend answers
  `409` with "no trained weights") is told so in one place: the section replaces
  both reads with a dashed callout that explains there is no attention to
  measure and links to the Luke brain, where **Train the brain** builds
  it. The reads, and their disabled lab buttons, are not drawn in that state.
- A read that fails for any other reason (the backend went away, the request
  was cut off) keeps its own error line and gains **Try again**, which asks the
  same two questions again from the same text and model. A reply that arrives
  after a newer read has started is dropped, so a superseded request can never
  paint an error over a fresh one. The lab's own **Predict again** button covers
  the same case inside the modal.

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

The **Tokenizer** card beside **Models** shows what the checkpoint was trained
with: the kind (word or byte-level BPE), the vocabulary count in its own unit,
the artifact id prefix, the characters it was built from, its tokens per
character, the artifact size, and which kind a new training uses. A missing or
changed BPE artifact shows the backend's error and a **Check again** action;
the card never substitutes a vocabulary. A word checkpoint says its vocabulary
lives inside the checkpoint, because it has no separate file.

### 5. Teach conversation expressions

`/dashboard/expressions` is the page for the words Luke uses at the edges of a
conversation. The trainer adds any wording under one of four moments — **open**,
**close**, **pause**, **resume** — and the reply it should get, then edits or
removes it from the same card. Each card shows the taught phrases with the
reply, the count against the limit, and an **Add expression** action.

- A taught expression is matched on the **whole message**, so "Bye" is answered
  by its own reply and "Bye, see you after lunch" is not captured by accident.
- A taught expression is answered before the built-in small talk, so the
  trainer's wording wins.
- The expressions also join the conversation training pairs, so the next
  training learns the same voice.
- The limits (120 expressions, 80 characters for the phrase, 240 for the reply)
  are shown on the page as the measured headroom; they come from the backend.
- Empty cards say that Luke uses its own wording for that moment.

The page reads and writes `GET/POST /language-model/expressions` and
`PUT/DELETE /language-model/expressions/{id}`.



The brain page shows the whole road from `0.0.0` to `1.0.0` as a game ladder:
101 rungs in ten bands, from Foundations to Judgment, ending at the Generalist.
A rung is marked **passed** when its level was approved, **ahead** when it was
measured and the approval is waiting, **next** only when it is the rung after
the approved level and a test for it exists, and **locked** otherwise. A locked
rung says that no test has been written for it yet and never shows a
measurement. The displayed level still moves only on the trainer's approval,
and measurement stops at `0.1.0` until the higher rungs have real tests.

### 6. Upload source documents

The **Documents** tab in `/dashboard/trainer` lists every file the assistant
answers from. Its first card is the upload tile: click it, or tab to its file
input and press Enter, to pick a csv, xlsx, docx, doc, pdf or txt file up to
`MAX_UPLOAD_BYTES`.

The request only hands the file over — `POST /upload` answers 202 with a job id —
so the tile never waits on a large read. While the job is queued or running the
tile shows its step and how far it has got ("Storing the passages · 96/312
passages") above a bar for `progress`, and two controls: **Stop indexing the
file** (`POST /ingest/jobs/{id}/cancel`, which removes the vectors the run had
already written) and, once a run failed or was stopped, **Retry indexing the
file** (`POST /ingest/jobs/{id}/retry`), which reads the stored file again from
the start. A finished job says "N passages stored, M skipped", where M is the
exact and near duplicates the collapse dropped, and a failed one shows the safe
one-line reason instead. Uploading bytes the workspace already holds is reported
as a duplicate and starts no job.

The progress comes from `GET /ingest/jobs/{id}/events`, a server-sent event
stream that repeats the job every half second until it reaches a terminal state;
the last twenty jobs are listed by `GET /ingest/jobs`. The refreshed file cards
follow the job with their chunk counts and view, download and remove actions.

### 7. Fine-tune a base model, then move its knowledge into Luke

The **Base model** tab runs one named training operation at a time on this
workspace's material and keeps every teacher it saves on this machine. It reads
`GET /language-model/huggingface` for the measured machine, the offered base
models, the saved teachers and the last run, and
`GET /language-model/huggingface/dataset` for the material.

- The machine chips show the device and the `torch` and `transformers`
  versions the backend measured.
- The dataset card says how many sources and examples the material holds and
  how many are training and held out for the selected operation. The split is
  taken per source, so a held-out source has no example on the training side.
  The card names the states that stop a run: nothing held out yet, held-out
  sources that also sit on the training side (listed by name), and no example
  for this operation. A dataset that could not be read is named in the card and
  in a toast, and the run button stays closed.
- **Base model** picks one of the offered checkpoints, with its parameter count
  and `saved` when a teacher for it exists, and shows the model's note.
  **Operation** picks **Supervised fine-tune (all weights)**, **Adapter
  (LoRA)** or **Continue pretraining**, and the note under it says what moves:
  a fine-tune replaces the saved teacher, an adapter is saved beside the
  teacher and promoted only if it beats the baseline, and continued pretraining
  reads uploaded documents as passages and changes nothing until documents are
  stored.
- **Steps** (1–2000, default 200) is the only run setting. The run button is
  disabled while a run is going, without a chosen model, or when the dataset
  cannot be trained on; the hint under it says which of those it is.
- The run card shows the operation, kind and teacher folder, the three stages
  (load the model, train, save) with their state, a bar that reaches 100 only
  when the run finished and the steps are done, the run line the backend
  recorded (steps, ETA, loss, material), the error when a run failed, and, for
  an adapter, the note that it is saved for measurement and not promoted: the
  chat still answers with the teacher.
- **Saved teachers** lists each teacher with its model, steps, loss, size and
  date, and removes one's weights after a confirmation. The last run report
  stays.
- Below the run card and the saved teachers, **Move the teacher's knowledge
  into Luke** asks one teacher every new question and stores its grounded
  answers as trained answers, so they survive the removal of the weights.
- The card picks a saved teacher and a question count (1–200, default 40) and
  says how many questions the material still offers, by source, with a sample
  list behind **Sample questions**.
- **Transfer into Luke** starts the run in the background; the card polls while
  it goes and shows the stage, the bar, how many were asked / kept / skipped,
  the last question and the answer it wrote.
- When a run is finished, **Undo transfer** removes exactly the answers that
  run created, after a confirmation, and leaves every other answer alone.
- The run is refused while a teacher is fine-tuning, and a fine-tune or a
  teacher removal is refused while a transfer is reading the teacher.

Below that card, **Download Luke** writes one zip with the model, the
tokenizers, the learner state and every material line. Before anything is
built it shows the model and level, the material count by source, the
checkpoint, the tokenizers and their size, the base checkpoint's identity, and
the file name with its size and file count for each mode; **Slim** and **Full**
pick whether the base checkpoint's weights travel with the package, and
**Download Luke** then builds the zip on the server and the browser saves it.
The card says that teacher weights are left out, and how many unanswered
questions ride along for context.

The same card restores a package: **Choose package** takes a zip,
**Check package** uploads it as a dry run and shows the file count and bytes
checked, the rows, the material and whether this workspace would have to be
replaced, and **Restore** applies it after a confirmation that says the current
workspace is replaced. A cancelled confirmation says nothing was changed; a
failed or incomplete report shows the reason and leaves the button and the
current workspace alone.

## The assessment page (`/dashboard/assessment`)

The assessment page is the only surface that measures Luke rather than teaching
him. It reads `GET /assessment/profiles?mode=…` for the eight cards and
`GET /assessment/runs` for the history, and nothing on it is trained or stored
as knowledge.

- **Frame**: one primary action in the top bar — the play icon that starts a
  run of the mode currently selected — and the info panel carries the page's
  title, its state and the four measured facts: held-out items and bank
  version, `done / total` scored in this mode, rubric answers awaiting review
  (warn tone while any wait) and how many runs are recorded across both modes.
  The run control, the stop control and the refresh control are icon-only with
  `aria-label` and `title`.
- **Modes** are tabs, not a toggle hidden in settings: *unaided* says the
  checkpoint answered with only what it can read in the task, *retrieval-assisted*
  says the workspace's own passages were offered first. The two modes are never
  blended into one number; switching tabs reloads that mode's latest run, its
  cards and its provenance.
- **Run panel**: the live row of the run — the step text the worker writes per
  item (`skill · item n of 240 · 6.4s`), a `done / total` count and a progress
  bar — plus the provenance a score needs: the checkpoint
  (`base@revision#weight-sha256`), the tokenizer's own file hash, the bank
  version and the held-out item count. Progress arrives over
  `GET /assessment/runs/{id}/events` as `event: run` frames, so the numbers move
  while the run works; a finished, failed or cancelled frame closes the stream
  and the page reloads the cards. A run that failed shows its `error` line under
  the provenance instead of a progress bar.
- **Skill cards** are one per skill, in the eight-skill grid: the score (or
  `not measured yet`, never `0`), the verdict, the stage, the confidence with
  the number of scored items behind it, the change against the same skill's
  previous run of the same mode, the topics never tested and the next step. A
  card whose skill has rubric answers waiting shows a `Review n waiting` button;
  a skill with nothing scored reads *Unassessed*. `app-card-info` states what
  the score is — the share of held-out items answered correctly in this mode, a
  stage benchmark, not an age or an intelligence claim.
- **Review** is one `app-modal` over the run's rubric rows for one skill: how
  many still wait, the item's topic and stage, the prompt, the answer the run
  stored (or that none was stored), the rubric's criteria under "What a good
  answer does" and the reference answer when the bank has one — then a 0–100
  score field and a note for the history. Saving posts
  `POST /assessment/results/{id}/review`, which marks that row scored with the
  reviewer's name and time, moves to the next row still waiting, and closes when
  the last one is decided; the card is rebuilt from the stored rows afterwards.
  The checkpoint never scores its own writing: an unreviewed rubric row keeps
  `score = null` and its skill stays unassessed.
- **Honesty**: a number on this page is only shown when a run produced it. With
  no run the cards read "not measured yet" and the panel says no run has been
  recorded; a request that fails shows the server's reason in a page-level
  alert, stamps the panel `not read` and says the skills could not be read —
  never that nothing was recorded, because that is not what is known. The grid
  is `repeat(auto-fit, minmax(240px, 1fr))`, so 390 px shows one full-width
  column with no page-level horizontal scroll, and the provenance values wrap
  rather than overflow.

The review dialog's two buttons sit in a `[modal-actions]` row, which is content
the page projects into `app-modal`; a global rule in `styles.css` gives every
such row the flex layout and the 8 px gap, because the modal's own stylesheet
cannot reach projected content. That rule covers the confirm dialog, the
expressions and design dialogs and the prediction lab the same way.

## Settings page

Settings is `/dashboard/settings`, reached from the settings control in the top
panel. It has six sections, in this order:

1. **Model identity** — the model's name (default `Luke`) and the version it
   starts from (default `0.0.0`). The name is used wherever the assistant
   speaks about itself, including the greeting, and the starting version is
   the floor of the level ladder. Saved through
   `GET`/`PUT /language-model/settings`. A reading that fails and a save that
   is refused are kept apart, so the card can say which happened: "The model
   identity could not be read: …" with a **Retry reading the model** action, or
   "The model identity was not saved: …".
2. **Foundation checkpoint** — the installed, verified base model (name,
   version, base model, revision, licence, parameters, context limit, weight
   size and dtype) and its runtime state (loaded or not, device, process
   memory). The card says the checkpoint is read-only and that a taught answer
   always wins. A missing or corrupt checkpoint shows the reason and a
   **Check the checkpoint again** action, never a made-up identity.
3. **Engine connection** — the backend URL and the live connection state.
4. **Live sources** — what the chat can answer from and where it falls back.
5. **Web evidence** — the allowlisted pages: the on switch, the search provider
   and address, one address field to add a source, the source list with its
   state and last check, **Check sources now**, and the counts of indexed,
   waiting and replaced pages. A failed request shows its message and
   **Try again**; the card never guesses a state.
6. **Appearance** — colour mode, text size, typeface and palette.

Each card has one info button; the explanations do not take up the page.

A web answer in the chat is labelled **Web page (unverified)**, shows
`as of <date>`, and its details panel carries the cited page and a link to it.
The composer's **Sources** menu has a **Web pages** switch beside weather and
Wikipedia; the per-question flag is what asks the backend for web evidence.

## Background tasks in the frame

A running background task is shown in two places: at the bottom of the open
primary sidebar, and pinned at the bottom of the open right sidebar. The app
root starts `BackgroundTasksService` once while signed in; it polls fast while
a task runs and slowly while idle, so a task that starts later still appears.
The right sidebar shows the task section only while a task runs; it never
replaces the page details or the engine information.

## When a read fails

- **The API layer turns a transport failure into words once.** Every service in
  `core/api` pipes every call through `BaseApiService.handleError`, which
  answers with the backend's own `detail` when there is one, the status when
  there is not, and "Could not reach the Luke Lab backend. Is the API running
  on …?" when the request never landed. A component then prints `error.message`
  and never an `HttpErrorResponse`: the transport's own message names the URL
  and ends in "0 undefined", which is a log line, not a sentence shown to a
  person. `core/api/error-mapping.spec.ts` holds the rule to one call per
  service, and a call added without the mapping is what breaks it.
- **The card names the thing that failed.** A bare reason is not enough when a
  page holds several reads: the model card says *the model identity* could not
  be read, the engine panel says *the engine* could not be read, and each keeps
  its reading failure and its refused write in separate signals so a save that
  was refused cannot be reported as a read that never happened.
- **A failed read is not an empty list.** An empty-state sentence is a claim
  about what the store holds, so it is suppressed whenever the read that fills
  that store failed: the trainer inbox does not praise an empty inbox, the
  trained-answer pills print `—` and the corpus facts `not read` instead of `0`,
  the "Showing x of y" line stays out, and the documents panel does not report
  an empty shelf. `store.error` was the first version of this and was worse than
  nothing: it was written by every failed read and rendered by no template, so
  the failure was invisible while the empty states kept asserting. The trainer
  now keeps `corpusError` and `documentsError` apart and names each in a strip
  above the tabs, with a button that reads that one list again.
- **A failed read is not a read in progress.** A placeholder that says
  "reading…" must clear when the read fails, not stand there forever: the
  settings fact strip and the engine version both stamp `not read` instead, and
  the assessment panel says the skills could not be read rather than that
  nothing was recorded.

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