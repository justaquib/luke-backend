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
├── start                     one-command backend launcher
├── scripts/                  startup, structure check, measurements
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

### Ingest pasted text (`POST /ingest`)

1. Client sends `{source, text}` to FastAPI.
2. `quality.pipeline.prepare_text` masks personal details when
   `PII_SCRUB_ENABLED` is on, and `nlp.chunk_spans` cuts the text:
   - Cleans whitespace and keeps the line breaks the sentencizer reads.
   - Loads `en_core_web_sm` (cached after first use).
   - Adds the `sentencizer` pipe if no parser is present.
   - Splits into sentences and packs them into passages (max 800 chars, 100-char overlap). A passage never starts or ends in the middle of a word, a sentence longer than the limit is split between words, and a line that starts a new question (`Q.`, `Question`) starts a new passage instead of finishing the previous one.
3. `dedupe.collapse` drops the passages the workspace already holds — the exact
   repeat by SHA-256 and the near repeat by 64-bit SimHash within five bits — and
   the repeats inside this text.
4. What is left is written as `documents` rows (with provenance, `language`,
   `quality` and `simhash`) and embedded into the `docs` collection with a stable
   id derived from the workspace, source and text.
5. Response: `{source, chunks, start_id, duplicates}`.

### Upload a file (`POST /upload`) and follow the job

A file is never read on the request thread. The route checks the extension, the
size (`MAX_UPLOAD_BYTES`) and that the bytes really are what the name claims,
refuses content the workspace already holds by SHA-256, writes the file under
`uploads/<workspace>/`, adds one `uploaded_files` row and one `ingest_jobs` row,
and answers **202** with `{job_id, source, status}`. A bounded in-process pool
(`INGEST_WORKERS`, default 2) then runs the job:

1. **Read** — `extraction.extract_document` opens the stored file and returns its
   pages with the character span each passage came from.
2. **Clean** — `quality.pipeline.prepare_document` removes the page furniture
   (headers, footers, page numbers and copyright lines that repeat across pages)
   and masks personal details when that is switched on.
3. **Cut** — `nlp.chunk_spans` splits the cleaned text into passages.
4. **Collapse** — `dedupe.collapse` drops the passages whose hash or fingerprint
   the workspace already holds, or that another passage in the same file already
   covered. The counts of what was skipped stay on the job row.
5. **Store** — `chunk_store.store_chunks` writes the rows and embeds the survivors
   `VECTOR_BATCH` (32) passages at a time, checking for a cancel before each
   batch. A cancelled or failed run removes the vectors it had written and leaves
   no `documents` rows behind; the file stays on disk so a retry can read it again.

`GET /ingest/jobs/{id}/events` repeats the job as server-sent events every 0.5 s
until it reaches a terminal state, `.../cancel` stops a running job, `.../retry`
queues a failed or stopped one again, and `GET /ingest/jobs` lists the last
twenty. Steps, the passages stored so far and the cancel flag live in memory
while the worker runs, so reporting progress never opens a second write
transaction; only the terminal state is written to the row.

### Query (`POST /query`) and the answer stream (`POST /query/stream`)

1. Client sends `{question, n_results}`.
2. `nlu.read_question` reads the question once: its words, each word's lemma and part of speech, the meaning and up to two synonyms of every key word from the offline dictionary, the question type and the key words. A dictionary that cannot be opened leaves the reading empty; it never fails the request.
3. `answer_flow.iter_answer_events` emits the stages as they happen: `reading`, then `thinking` (the question type and the sources it plans to consult), then one `connecting` per source as it is consulted, and finally `answer`. `POST /query` consumes the same iterator, so the JSON and the stream cannot drift.
4. `vector_store.search_faq` looks for a trained answer in the `faq` collection. If the best match has a cosine distance ≤ `FAQ_MATCH_THRESHOLD` (default `0.28`), the trained answer is returned with `source: "faq"`.
5. Otherwise a loose FAQ match within `INTENT_MATCH_THRESHOLD` is returned with `source: "intent"`.
6. Otherwise the shared taught-answer path (`answering.answer_from_curriculum`) answers: conversation first (a greeting, the model's name from Settings, thanks, a goodbye, small talk; `source: "assistant_profile"`), then the learner's lessons and quick facts. This is the same path the learner page uses, so a question cannot be answered one way in the chat and another way in the learner.
7. Otherwise the live sources answer when they are enabled: weather, then Wikipedia.
8. Otherwise `vector_store.query` falls back to the `docs` collection. It fetches `max(4 × n_results, 8)` semantic candidates, ranks every stored chunk of the workspace by BM25 (`retrieval.bm25_scores`, k1 1.5, b 0.75, stopwords dropped), and fuses the two rankings by reciprocal rank (`retrieval.fuse_rankings`, k = 60), so a chunk both searches rank high climbs above a chunk only one of them found. The best `max(5 × n_results, 20)` fused candidates are optionally re-read by a cross-encoder (`rerank.py`, off unless enabled) and the top `n_results` are returned.
9. Otherwise, when the foundation checkpoint is installed, `foundation/answers.py` writes the answer from the base model. The answer is labelled `mode: "generated"` and unverified, carries no citation, and is never written to lessons, FAQs or training pairs. A composed request (`compose: true`) numbers the retrieved passages, requires citations and runs the verifier; the passages are served instead when the check fails.
10. If the checkpoint is missing or fails while writing, the question is logged to `pending_questions` (occurrence counter incremented if it was already there) and a "trainer has been notified" placeholder is returned with `source: "pending"`. A missing checkpoint is reported, never replaced with an invented answer.
11. Every answer is finished once (`language/finishing.py`): outer whitespace trimmed, the first letter capitalized, and a full stop added only when the text ends in a letter or digit. Words are never reworded. The answer carries an optional `understanding` (words read, plan, connections, grammar notes), and `nlp.extract_entities` runs spaCy NER over the question for the frontend to display.
12. Response: `{question, source, mode, answer, hits, citations?, entities, understanding, model, model_name, pending_id?}`.

`POST /query/stream` returns the same answer as a server-sent event stream: `reading`, `thinking`, one `connecting` per source, `writing` and one `token` per generated piece when Luke writes, `answer` (the full JSON above), then `done`. An `error` frame is terminal and no `done` follows it. The caller's token is checked before the stream opens, so 401/403 stay normal HTTP responses.

### Reading and connecting a question

`app/services/nlu/` reads the question before anything is looked up. One spaCy parse gives each word its lemma and part of speech (the blank pipeline still works), `reasoning.entities.entities_of` picks the key words, `dictionary.lookup_word` gives the first definition of each key word and `dictionary.search_expansions` up to two synonyms, and `reasoning.intent.shape_kind` names the question type.

`app/services/answer_flow.py` owns the decision path. Each finder returns a `SourceStep` whose `Connection` records the source, an outcome (`match`, `miss`, `pass`, `fail` or `skipped`), the measured detail (a distance, the lesson evidence, "not enabled", "no matching page") and the question that matched. A stage is never invented: the page shows exactly the connections the backend produced, and the same list is stored with the answer and shown in its details.

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
| `GET /language-model/settings`         | The model's name and the version it starts from            |
| `PUT /language-model/settings`         | Save the model's name and starting version                 |
| `GET /language-model/expressions`      | The taught opening, closing, pause and resume lines        |
| `POST /language-model/expressions`     | Teach one expression and the reply it gets                 |
| `PUT /language-model/expressions/{id}` | Edit a taught expression                                   |
| `DELETE /language-model/expressions/{id}` | Remove a taught expression                              |
| `GET /language-model/foundation`       | The installed checkpoint's verified identity and runtime state |
| `POST /language-model/brain/correction` | Teach the right answer to a question answered wrongly    |
| `POST /language-model/brain/train`     | Train the brain on the tenant's own material               |
| `POST /language-model/lessons/check`   | Report what the learner already knows about a lesson       |
| `GET /language-model/lessons/duplicates` | The lessons that repeat each other, grouped for review    |
| `POST /language-model/lessons/duplicates/accept` | Keep every lesson of a group: the repeat is on purpose |
| `POST /language-model/lessons`         | Teach a lesson, or a story, a lesson or a topic            |
| `PUT /language-model/lessons/{id}`     | Edit a lesson the learner already holds                    |
| `DELETE /language-model/lessons/{id}`  | Remove a lesson                                            |
| `POST /language-model/preview`         | Answer a question with the learner's own material only     |
| `POST /language-model/preview/stream`  | The same answer, read and connected stage by stage         |
| `POST /language-model/test/start`      | Start a mock test, a final exam, or a trainer led test     |
| `POST /language-model/test/question`   | Ask the learner a question during a grade test             |
| `POST /language-model/test/score`      | Score the learner's last answer                            |
| `POST /language-model/test/finish`     | Close the test and open the next grade when it is passed   |
| `GET /language-model/dictionary`       | Look a word up in the offline dictionary                    |
| `POST /language-model/dictionary/learn`| Teach the learner a word and the meaning it should keep    |
| `POST /language-model/sentence`        | Build a sentence from the words the learner knows          |
| `POST /parenting/chat/session`          | Start a coaching session between the parent and the coach |
| `GET /parenting/chat/session/{id}`      | Read one coaching session                               |
| `POST /parenting/chat/session/{id}/turn` | The coach answers the parent's turn from the taught lessons |
| `POST /parenting/chat/session/{id}/turn/stream` | The same turn, read and connected stage by stage     |

Every route needs the same bearer token as the rest of the API and only ever
reads or writes the tenant of the signed in user.

### The model choice and the Hugging Face teachers

`app/routers/huggingface.py` is thin; `app/services/model_choice.py` decides
which model answers a message and `app/services/huggingface/` owns the teachers
(catalogue, files, fine-tune, prompt and serving). A teacher is fine-tuned on
the tenant's own taught material, written under
`.language_model/huggingface/tenant-<id>/teacher-<model>/`, and loaded once and
kept while it answers. `transfer.py` moves that teacher's knowledge into Luke
and `seeds.py` builds the questions it asks, so transfer can be tested and
sized without loading a model. `language/package/` writes Luke himself to one
zip and reads one back: `bundle.py` (preview and build), `manifest.py` (the
format, the hashes and the checks), `dataset.py` (the rows as JSON lines),
`reindex.py` (rebuild and verify the collections) and `restore.py` (staging,
validation, the transactional replace and the verification); `export.py` is the
thin facade the routes call. A package is format 2 and holds `manifest.json`,
`README.md`, `weights/checkpoints/`, `weights/tokenizers/<id>/`, `weights/base/`
(full mode only), `state/`, `dataset/*.jsonl` and `raw/`. The manifest records
the checkpoint and tokenizer identities, the licence, the revisions, the
embedding model, the collection settings, the counts, and the size and SHA-256
of every file. Teacher weights stay out of it.

| Route | What it does |
| --- | --- |
| `GET /language-model/models` | Luke and every installed teacher, with the default |
| `POST /language-model/predict` | The next words and the measured attention of the chosen model |
| `GET /language-model/huggingface` | The machine, the offered base models, the saved teachers and the last run |
| `POST /language-model/huggingface/train` | Fine-tune one base model in the background (one run at a time) |
| `DELETE /language-model/huggingface/teachers/{folder}` | Remove one teacher's weights |
| `GET /language-model/huggingface/transfer` | The questions a transfer would ask, the run in progress and the last report |
| `POST /language-model/huggingface/transfer` | Ask one teacher every new question in the background |
| `POST /language-model/huggingface/transfer/undo` | Remove the answers the last transfer created |
| `GET /language-model/export/preview` | What the downloadable package would hold, counted before it is built |
| `GET /language-model/export` | Build the zip, send it, and delete the temporary file |
| `POST /language-model/package/import` | Dry-run or restore a package into this workspace, from an uploaded zip |

An import writes nothing until the manifest, every file hash, the counts, the
row types and the local references have been checked, and it defaults to a dry
run that reports what it would do. Restoring into a workspace that holds rows
also requires `replace=true`; the replace is transactional, so a failure puts
the old rows, files and collections back. The zip is never extracted by path:
member names are checked against traversal and the archive and extracted sizes
are capped. A restored index is built again from the database with the
embedding and collection settings the manifest names, then the counts and a
document and FAQ sample are read back before the import reports success. A
package whose workspace number is not the caller's is refused, because a
package cannot change workspace.

The chosen model is sent with `/query` and `/query/stream`. A teacher writes
its answer from the retrieved reference (the matched answer plus the retrieved
passages) and must share enough of the reference's words to be shown; when it
does not, the matched answer is served and the `model` connection records the
fallback. A teacher's measured limits are in `docs/07-model-measurements.md`;
the chat always labels the answer with the model that wrote it.

A session lasts a week. `POST /auth/refresh` gives a signed-in user a new
token, so the app renews the session while it is in use instead of asking the
user to sign in again every day.

A lesson is a question and an answer. A story, a lesson or a topic is taught
as one passage instead: it is filed under the name it was given, every
sentence of it is taught on its own as well, and the learner can ask for it
by that name or by its topic.

The model's identity is tenant state too: `model-<tenant>.json` keeps the name
(default `Luke`) and the starting version (default `0.0.0`). The name is the
one the assistant uses when it speaks about itself, and the starting version
is the floor of the displayed level ladder. The greeting pairs are built from
the same name, so a rename changes the voice and the training together.

The whole ladder is data, in `app/services/brain_roadmap.py`: 101 rungs from
`0.0.0` to `1.0.0`, grouped into ten bands and ending with the Generalist. A
rung is marked `open` only when `LUKE_LEVELS` holds a real test for it, so the
page can show the full road while measurement stays at the last tested level
(`0.1.0`). `GET /language-model/brain` returns every rung with its band and
open state; the page locks every rung that is not the next open one, and the
displayed level still moves only on the trainer's approval.

Training the brain takes its time, so it runs in its own process: the app
stays usable while the model learns. The tenant's learner file carries a
`training-<tenant>.json` marker while that runs, and the page asks the brain
again until the marker is gone. One checkpoint is written next to the model
(`words-<tenant>.pt` for next-word prediction), and the page picks it up as
soon as the file changes, so the vocabulary grows with every lesson that was
taught.

### The foundation checkpoint (Luke 0.0.1)

`app/services/foundation/` owns the pretrained checkpoint. It is pinned,
read-only and verified against a recorded SHA-256; nothing in the product
writes to it or trains it in place. The modules are:

| Module | What it owns |
| --- | --- |
| `identity.py` | The pinned record (name, version, base model, revision, licence, parameters, context limit, weight hash and size) and the status the settings page reads. A missing or corrupt checkpoint is an actionable error, never a fabricated answer. |
| `runtime.py` | Loading and serving: one cached model per process, float16 on MPS and float32 on CPU, the checkpoint's own chat template with a documented fallback, configurable generation, token-by-token streaming, cancellation and peak-memory reporting. |
| `prompt.py` | The system prompt, bounded conversation history, and numbered evidence when the caller asks for a composed answer. |
| `verify.py` | The grounded check: every citation resolves to supplied evidence, and no number or named entity appears that the evidence does not support. |
| `answers.py` | The writing path. It runs only after every taught source missed, labels the answer `generated` (or `hybrid` when composed from numbered passages), and never stores it. A failed check falls back to the passages themselves. |

The chat reports `mode`: `retrieved`, `generated` or `hybrid`, plus the model
name and version. Only a `retrieved` answer is taught material; `generated`
carries no citation and is unverified; `hybrid` cites numbered passages and is
verified against them. The stream adds `writing` and `token` events, stops on
client cancel, and finishes with the same final answer the JSON route returns.
The measured speed, memory and perplexity of the installed checkpoint are in
`docs/07-model-measurements.md`.

### The neural engine

`app/services/neural/` is the one place that owns the models. Nothing outside
the package imports `torch` or opens a checkpoint file:

| Module | What it owns |
| --- | --- |
| `engine.py` | `NeuralEngine`: the word task of one tenant, its training, its saved file and the numbers the pages read. `engine_for(tenant_id)` keeps one engine per tenant and reloads it when the checkpoint changes. |
| `words.py` | `WordTask`: plain next-word prediction over the tenant's sentences, read and written through the tokenizer its checkpoint was trained with. |
| `writer.py` | The generation rules: a mark is never written, a word model does not say a naming word twice, and a BPE model's pieces are decoded and trimmed to a sentence end. |
| `prediction.py` | The next-token rows and the attention heatmap, labelled in the tokenizer's own units (words or pieces). |
| `tokenizer/` | The scratch tokenizers: `base.py` (interface and errors), `word.py` (the legacy word adapter, byte-for-byte ids), `bpe.py` (byte-level BPE), `corpus.py` (passage provenance and the leakage-free split), `artifacts.py` (content-addressed artifacts), `spec.py` (which kind a new training uses). |
| `training.py` | The training loop: a group split, fixed train and held-out batches, warmup then cosine, early stopping on the held-out loss, a wall-clock budget, and the report the page reads. |
| `measurement.py` | The held-out split, the fixed reads, and the chance loss a model is compared against. |
| `checkpoint.py`, `store.py` | Schema v3 checkpoint files (which add the `tokenizer` record), written atomically, plus the short run history the builder shows. |
| `serving.py` | The word surface the language services use; it reports the backend and the quality numbers behind it. |

The models are mapped from disk rather than copied into memory: the checkpoint
is read with `mmap=True` and its weights are assigned into a model skeleton, so
the pages that only show numbers never build a model at all, and a request
pays only for the task it uses. One engine is kept at a time, because a local
install serves one tenant.

### The word model

The served model is the word model: plain next-word prediction over the
tenant's sentences, used by the learner's sentence feature. It replaced the old
trigram, and when a tenant has not trained one yet, the learner page says so
and produces no predictions instead of falling back to n-gram guesses. Its
checkpoint and its held-out loss stand on their own, so the page reports the
quality of the model that actually runs.

The answer is written by a model only when no taught material answers the
question. A taught answer, an approved answer, a lesson or a dictionary entry
is served as it was taught and always wins. When every taught source misses,
the foundation checkpoint writes an answer that is labelled written and
unverified; that answer is never stored as knowledge.

### The scratch tokenizer

A tokenizer belongs to the weights it was trained with, and three id spaces are
kept apart: the legacy word-level vocabulary, the BPE vocabulary, and each
pretrained model's own tokenizer (Luke 0.0.1 and every teacher). They are never
mixed, remapped or silently substituted.

- A schema-3 checkpoint records `tokenizer: {kind, id, vocab_size,
  special_ids}` and still carries `vocabulary` for the legacy adapter, so a v1
  or v2 checkpoint loads through the word tokenizer unchanged.
- BPE is byte-level, NFC-normalised, with fixed special ids 0–4; a line break
  is one `<br>` mark so a poem keeps its shape, and a raw encode never produces
  a special id (text that spells `<pad>` stays text).
- A BPE artifact is named by the SHA-256 of its `tokenizer.json` under
  `.language_model/tokenizers/{id}/` and is verified by hash before use. A
  missing or changed artifact is an error the page shows, never a fallback.
- The measured experiment kept the word tokenizer as the default (see
  `docs/07-model-measurements.md`); BPE is opt-in per training.
- The corpus is built from the training side of the group split only, so a
  held-out sentence can never shape a vocabulary.


### What the brain trains from

The training is built from every source the trainer has given, and they are
collected in one place, `app/services/language/training_materials.py`:

| Source | What it is |
| --- | --- |
| Lessons | The question and the answer of every lesson in the learner, and the starter lessons of the grades it reached. |
| Conversations | The greetings, the model's name, thanks, goodbyes and small talk, built from the name in Settings. |
| Approved answers | The questions and the answers the trainer approved in the Trainer. |
| Corrections | The answers the trainer corrected, so the model stops making the same mistake. |
| Documents | The chunks of every uploaded document. |
| Words | The words and the meanings that were taught with the dictionary. |
| Sentences | The sentences that were taught on their own. |

Every source becomes a sentence, so the words of it reach the vocabulary. The
answers of the lessons, approvals, corrections and conversations are also
collected in the wordings they can be asked for, so the words a trainer would
use for a subject are in the corpus too. The word model learns from that text
alone.

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

A definition question that names no lesson title can still be answered by a
lesson taught in other words. `app/services/language/fallback.py` asks the
offline dictionary for up to six WordNet synonyms of the question word and
looks each one up with the same lesson matching. The fallback answers only
when **exactly one** distinct lesson is reached and the question shares a
stemmed key word with that lesson's title or answer; a synonym that reaches
nothing, or two lessons, leaves the dictionary answer in place. Multi-word
synonyms are not tried, because a phrase matches too loosely to name a lesson.
The connection is recorded as `fallback` with `pass` or `fail`, so the reading
trace shows why a synonym did or did not place the lesson.

### Answering a request, not a question

A lesson is taught as a question and an answer, so a lookup finds it. A
request is different: "write a short poem about the rain" shares almost no
words with the lesson "Poem: Rain Rain Go Away", so no amount of question
matching finds it.

The brain answers requests in three steps.

| Step | Where | What happens |
| --- | --- | --- |
| Read the request | `app/services/language/shapes.py`, `requests.py` | The request is matched against the shapes it knows: a poem about a topic, a list of a topic, some facts about a topic. The topic is taken out of it. |
| Find the material | `app/services/language/topics.py`, `requests.py` | The topic index is counted over the lessons of the kind the request asks for, so a word shared by many fact lessons can still name the one poem. A word that many lessons *of the same kind* share names none of them, and the assistant refuses rather than choosing one at random. |
| Serve the lesson | `app/services/language/requests.py` | The lesson the request named is returned as it was taught, marked `taught`. The words are the trainer's, never rewritten. |

A request about something never taught is never answered with an invented
answer: the material boundary returns no answer and the chat answers that a
trainer has been notified. Exact matches keep their order — FAQ, intent,
taught material, weather, wiki, documents — and the first one that has an
answer wins.

### The system administrator in the chat

The system administrator has no tenant of their own: they look after every
tenant instead of belonging to one. The chat answers them anyway, reading the
material of the tenant-less bucket (where the lessons they taught themselves
live) and, with no tenant filter on it, the approved answers and the
documents of every tenant they look after. Their own uploads and feedback land
in that tenant-less bucket (`tenant_id NULL`), so the material they add is
platform-owned rather than filed under a tenant. A user without a tenant is
still turned away, because they have no assistant of their own.

### Correcting an answer

`POST /language-model/brain/correction` takes the question, the answer that
was given and the answer that should have been given. It:

- stores the right answer as a trained answer of the tenant, so the next
  question of that kind is answered correctly from the FAQ collection,
- keeps the correction in the brain's own memory, so the heads learn from it,
- asks the question again and checks that the assistant keeps the correction,
  so the trainer is told whether it really learned it.

### The level test

`GET /language-model/brain?force=true` measures the rungs that have a real test
and keeps the result. The measure uses the generality probes: held-out
questions the training never showed, so a model that only memorises scores
nothing on them. The score is in `model_level.generality`, the probes are in
`model_level.generality_probes`, and the level itself still moves only on the
trainer's approval.

## Web evidence (Phase 4)

The web is optional evidence, not a trained source. It is asked for per
question (`QueryRequest.web`), only after the taught material and the documents
have had their turn, and only when the workspace policy allows it.

- **Policy** (`app/services/web/trust.py`) is stored per workspace as
  `.language_model/web-<n>.json`: on or off, provider (`searxng` by default,
  `brave` or `tavily` optionally), the SearXNG address, the allowlist, the time
  to live, the minimum similarity and how many results to read. A search API
  key is read from `WEB_SEARCH_API_KEY` only; it is never written to the policy
  file, the database or the browser.
- **Trust tiers**: a page on an allowlisted domain gets the `official` tier and
  may be indexed and quoted; any other page is `general` and becomes a pending
  FAQ candidate in the trainer's inbox. It is never indexed, so no intent or
  FAQ lookup can serve it, and nothing is trained automatically.
- **Guarded reader** (`fetch.py`) refuses a non-http(s) address, credentials in
  the URL, a port other than 80/443, a local name, and any address that
  resolves to a non-public IP — re-checked on every redirect. It reads
  `robots.txt` first (longest match wins, allow wins a tie; a 4xx allows, a
  5xx or unreadable file disallows), caps redirects and body size, keeps two
  reads of one host 0.5 s apart, accepts only `text/html`,
  `application/xhtml+xml` and `text/plain`, and extracts the readable text with
  trafilatura. Raw markup is never stored.
- **Snapshots** (`web_snapshots`) record the URL, title, text, a SHA-256 of the
  text, the tier, the fetch and check times and a status. The same text re-read
  moves the dates forward; changed text supersedes the old row and re-indexes;
  a superseded row is kept for citations until the time to live expires it.
- **Index**: an official snapshot is chunked (at most 24 passages) into the
  `web` collection with `verified: "false"` and an integer `fetched_at`, so a
  search only sees passages inside the time to live.
- **Answer**: the step first searches the stored passages, then — only on a
  miss — searches, reads and stores. A hit answers with its URL, the date it
  was fetched (`as_of`), a citation and a connection that says it is unverified
  and not taught material. Only general results leave the web step a miss.
- **Refresh**: `scripts/refresh_web.py` re-reads the allowlisted sources that
  are due, records `checked / changed / unchanged / failed / expired`, and
  removes expired snapshots and their vectors. Run it from cron; a daily run
  suits the default seven-day time to live.
- **Routes**: `GET /web/status`, `PUT /web/settings`, `POST /web/sources`,
  `DELETE /web/sources/{id}`, `POST /web/refresh`.

## The assessment engine (Phase 6)

The engine answers a fixed, licensed item bank with the installed checkpoint and
marks what comes back. It measures Luke; it does not teach him. Nothing an
assessment run produces is stored as knowledge, and no answer is ever executed.

- **Skills** (`app/services/assessment/skills.py`) — eight profiles, one per
  measured ability: grammar, conversation, reading, creative writing,
  mathematics, programming, general knowledge and science. Each is banked at
  three stages with `EVAL_ITEMS_PER_SKILL = 30` held-out items and
  `TRAIN_ITEMS_PER_SKILL = 15` items a trainer may use.
- **Bank** (`bank.py`, `app/data/assessment/<skill>.json` and `manifest.json`) —
  the items are vendored offline by `scripts/build_item_bank.py` from four
  licensed datasets (AI2 ARC, BLiMP, Databricks Dolly, Google IFEval), each
  file recording its own dataset and licence. The manifest carries the version,
  the split salt, the source fetches and a SHA-256 per file; `bank.check()`
  re-reads every item through `intake.validate_item` and reports any problem, so
  a malformed item raises instead of being scored and no run touches the network.
- **Answer kinds** (`scoring.py`) — `option`, `span`, `number`, `literal` and
  `rules` are marked in code; a `rubric` is answered and then waits, its score
  `None`, which is unassessed and never zero. Each scorer normalises the way a
  person reads an answer — case, articles, punctuation, surrounding prose, a
  value inside backticks or a fence — and then compares exactly. A literal goes
  through `ast.literal_eval` and nothing else; `PASS_MARK = 0.8` is what an
  outcome must reach to count as cleared.
- **Profiles** — `skills.build_profile` turns a skill's outcomes into a score,
  a stage, a confidence (`high` at 20 scored items, `medium` at 10, `low` below
  that, `none` with nothing scored), strengths and weaknesses per topic, the
  topics never tested, the next step and the delta against the same skill's
  previous run of the same mode. A stage only clears when
  `STAGE_MIN_ITEMS = 3` items of it are scored, so one lucky answer cannot
  promote a skill.
- **Modes** (`conditions.py`) — `unaided` builds the message from the task
  alone, and the evidence step exists only in the `assisted` branch, which
  prepends up to `MAX_EVIDENCE_ITEMS = 4` passages of `MAX_EVIDENCE_CHARS = 600`
  characters. The same flag is honoured by every source step of the request
  path, so an unaided score cannot be quietly lent an answer. Generation is
  greedy (`temperature = 0.0`, `max_new_tokens = 220`) so a number does not move
  with a sampling seed.
- **Run** (`runner.py`) — a background job on a one-worker pool, like an upload.
  Each run row records the provenance a score needs to mean anything: the
  checkpoint name and version, `base@revision#weight-sha256`, the tokenizer's
  own file hash, the generation settings as JSON, and the bank version and salt.
  Every item writes its result and its `latency_ms` as it arrives, publishes
  `done / total / step` to the row and to the SSE stream, stops between items on
  a cancel, and `recover()` marks a run a restart interrupted as failed while
  keeping the results already stored.
- **Review** (`profiles.py`) — a trainer's score for one rubric row writes
  `score`, `reviewed_by`, `reviewed_at` and a note, and the skill's card is
  rebuilt from the stored rows. The 0.5B checkpoint never judges its own
  writing; a rubric skill stays unassessed until a review exists.
- **Leakage** (`integrity.py`) — no `eval` item may reach a training loader, the
  tokenizer corpus or stored material. `is_held_out` drops a sentence before it
  is kept (it guards `language/writing.py`, `language/corpus.py` and
  `language/training_materials.py`) and `scan` reports stored text that matches
  a held-out prompt. The ban is unconditional; `tests/test_assessment_integrity.py`
  and `tests/test_assessment_loader.py` assert it.
- **Tables** — `assessment_runs` (mode, status, progress, provenance, error and
  the three timestamps) and `assessment_results` (one item answered: the response,
  the marking method, score, `correct`, `latency_ms`, the rubric fields and the
  split, skill, stage and topic copied from the bank so a run reads without
  re-opening it). `scripts/migrate_assessment.py` creates them with a backup, a
  dry run and a rollback.
- **Routes** — `GET /assessment/profiles`, `POST /assessment/runs`,
  `GET /assessment/runs`, `GET /assessment/runs/{id}`,
  `GET /assessment/runs/{id}/events`, `POST /assessment/runs/{id}/cancel`,
  `GET /assessment/runs/{id}/results`, `POST /assessment/results/{id}/review`.

## Data model

### ChromaDB collections

- `docs` (configurable via `CHROMA_COLLECTION`)
  - **id**: UUID5 of `"<workspace>:<source>:<sha256 of the passage>"`, so
    re-indexing the same passage overwrites instead of duplicating
  - **document**: string, the passage text
  - **metadata**: `{ "source": "notes", "tenant_id": 0, "page_number",
    "document_id", "chunk_index", "char_start", "char_end", "content_hash",
    "language", "quality" }` — only the fields the file actually provided

- `faq` (configurable via `CHROMA_FAQ_COLLECTION`)
  - **id**: string of the `faqs.id` primary key
  - **document**: the question
  - **metadata**: `{ "answer": "...", "faq_id": 42 }`

- `intents` (configurable via `CHROMA_INTENTS_COLLECTION`)
  - **id**: string of the `intents.id` primary key
  - **document**: the intent name, the wording the trainer taught
  - **metadata**: `{ "response_template": "...", "faq_id": 42 }`

- `web` (configurable via `CHROMA_WEB_COLLECTION`)
  - **id**: UUID5 of the workspace, URL and text hash
  - **document**: one readable passage of an allowlisted page
  - **metadata**: `{ "source": url, "url", "title", "trust": "official",
    "verified": "false", "fetched_at": <epoch int>, "snapshot_id",
    "content_hash" }`

All four are stored on disk in `./.chroma/`. Delete the folder to start over.
Every collection is created with `cosine`, `hnsw:M=16`,
`hnsw:construction_ef=200` and `hnsw:search_ef=100` (`chroma.COLLECTION_CONFIG`),
which is twice and ten times Chroma's own defaults; the settings are read back
with `chroma.collection_settings` whenever a measurement reports what the search
actually used. The stale `lessons` collection (651 vectors, no metadata, written
only by the pre-Phase-2 trainer) was retired on 2026-10-28 with
`scripts/retire_artifacts.py`, which backed it up to
`backups/retired-<timestamp>/lessons.json` before removing it; the same run moved
the unused `transformer-0.pt` checkpoint into that folder.

### Where the vectors are read fast

Three caches sit between a request and the store, and all three are bounded:

- **Query-embedding cache** (`services/embeddings.py`): the collections are
  opened with an embedding function that keeps the vectors of short texts (512
  entries, at most 256 characters each). The same question asked twice does not
  reach the model twice; a long uploaded passage is embedded but not kept.
- **Corpus cache** (`services/corpus_cache.py`): the whole-workspace read used
  for the BM25 half of the ranking, with its word counts already built, is kept
  until `chroma.sqlite3` changes its modification time. At most four workspaces,
  and none larger than two million characters, are held.
- **Model cache** (`services/model_cache.py`): the word engine and the Hugging
  Face teacher are kept by `ByteBoundedCache`, which drops the least recently
  used model once the weight files pass `WEIGHTS_CACHE_BYTES` (1.5 GiB) instead
  of keeping exactly one model. A model is never evicted while it is the only
  one held, so a single workspace cannot thrash.

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

The response carries a new `app` payload and a `source` of `weather`, `wiki`
or `web` so the UI can render source tags and links. Web answers also carry
`as_of`, the date the quoted page was fetched.

### Web tables

`web_sources` — one allowlisted page per row, with its domain, workspace and
whether refresh should read it.

| Column       | Type      | Notes                                  |
| ------------ | --------- | -------------------------------------- |
| `id`         | SERIAL    | Primary key                            |
| `url`        | VARCHAR   | The page the trainer added             |
| `domain`     | VARCHAR   | The site the allowlist entry names     |
| `enabled`    | BOOLEAN   | Whether refresh reads it               |
| `tenant_id`  | INTEGER   | The workspace                          |

`web_snapshots` — every read of an allowlisted page, keyed by its content hash.

| Column         | Type      | Notes                                        |
| -------------- | --------- | -------------------------------------------- |
| `id`           | SERIAL    | Primary key, also `snapshot_id` in Chroma    |
| `source_id`    | INTEGER   | The source it was read for, when known       |
| `url`          | VARCHAR   | The page address                             |
| `title`        | VARCHAR   | The page title                               |
| `text`         | TEXT      | The readable text, never raw markup          |
| `content_hash` | VARCHAR   | SHA-256 of `text`                            |
| `trust`        | VARCHAR   | `official` or `general`                      |
| `status`       | VARCHAR   | `indexed`, `candidate` or `superseded`       |
| `fetched_at`   | TIMESTAMP | When this text was read                      |
| `checked_at`   | TIMESTAMP | When the page was last checked               |
| `tenant_id`    | INTEGER   | The workspace                                |

The first table is what refresh walks; the second is the reproducible evidence a
web answer cites.

### Postgres tables

`documents` — one row per stored passage, with where it came from and what it is worth:

| Column          | Type      | Notes                                              |
| --------------- | --------- | -------------------------------------------------- |
| `id`            | SERIAL    | Primary key                                        |
| `source`        | VARCHAR   | Logical source name                                |
| `chunk_index`   | INTEGER   | Position within that source                        |
| `text`          | VARCHAR   | Passage text                                       |
| `tenant_id`     | INTEGER   | The workspace                                      |
| `created_at`    | TIMESTAMP | Default `now()`                                    |
| `document_id`   | INTEGER   | The `uploaded_files` row the passage was read from  |
| `char_start`    | INTEGER   | First character of the passage in the extracted text |
| `char_end`      | INTEGER   | Last character of that passage                     |
| `page`          | INTEGER   | Page it was found on, when the file has pages      |
| `content_hash`  | VARCHAR   | SHA-256 of the passage text — the exact-duplicate key |
| `simhash`       | BIGINT    | 64-bit SimHash of its words — the near-duplicate key, within five bits |
| `language`      | VARCHAR   | `en` or `unknown`, from `language/identify.py`      |
| `quality`       | FLOAT     | `language factor × text density × (1 − repetition)`, 0–1 |

Rows written before a column existed are filled by a script, never by a silent
rewrite at request time: `scripts/migrate_simhash.py --dry-run` then `--yes`
fingerprints the passages without one, and `scripts/migrate_quality.py --dry-run`
then `--yes` measures the language and the quality the ingest path would have
given them. Both write a report under `backups/` and leave a row that already
has a value alone.

`uploaded_files` — one row per file kept on disk:

| Column         | Type      | Notes                                          |
| -------------- | --------- | ---------------------------------------------- |
| `id`           | SERIAL    | Primary key, also `documents.document_id`       |
| `source`       | VARCHAR   | The name it was uploaded under                 |
| `content_hash` | VARCHAR   | SHA-256 of the bytes; a repeat is refused, not re-read |
| `size_bytes`   | INTEGER   | What the file weighs                           |
| `tenant_id`    | INTEGER   | The workspace                                  |
| `created_at`   | TIMESTAMP | When it arrived                                |

`ingest_jobs` — the background read of one upload, and what the page polls:

| Column              | Type      | Notes                                              |
| ------------------- | --------- | -------------------------------------------------- |
| `id`                | SERIAL    | Primary key, named in the `POST /upload` answer     |
| `source`            | VARCHAR   | The file it is reading                             |
| `status`            | VARCHAR   | `queued`, `running`, `done`, `duplicate`, `failed`, `cancelled` |
| `step`              | VARCHAR   | The stage the worker reported last                 |
| `progress`          | FLOAT     | Share of the passages stored so far                |
| `done` / `total`    | INTEGER   | Passages stored, passages to store                 |
| `chunks`            | INTEGER   | Passages the run wrote                             |
| `exact_duplicates`  | INTEGER   | Passages dropped for the same SHA-256              |
| `near_duplicates`   | INTEGER   | Passages dropped for a SimHash within five bits    |
| `error`             | VARCHAR   | A safe one-line failure, never the traceback       |
| `tenant_id`         | INTEGER   | The workspace                                      |
| `created_at` / `started_at` / `finished_at` | TIMESTAMP | When the job was queued, ran and ended |

A job interrupted by a restart is marked retryable on startup
(`ingest_jobs.recover()`), and the live step and counts stay in memory while it
runs, so progress costs the worker no extra write.

`faqs` (trained Q↔A, written by the trainer or by the inline ✏ correction):

| Column        | Type          | Notes                                  |
| ------------- | ------------- | -------------------------------------- |
| `id`          | SERIAL        | Primary key, also used as Chroma id    |
| `question`    | TEXT          | The canonical question                 |
| `answer`      | TEXT          | The trained answer                     |
| `created_at`  | TIMESTAMP     | Default `now()`                        |
| `updated_at`  | TIMESTAMP     | Updated on edit, mirrored to Chroma    |

`pending_questions` (questions the bot could not answer, and answers that
contradict one the trainer already gave, awaiting the trainer):

| Column        | Type          | Notes                                  |
| ------------- | ------------- | -------------------------------------- |
| `id`          | SERIAL        | Primary key                            |
| `question`    | TEXT          | The unresolved question                |
| `note`        | TEXT          | Why it is in the inbox: the two answers that disagree and which one is kept |
| `occurrences` | INTEGER       | How many times it was asked            |
| `resolved`    | BOOLEAN       | True once a trainer provides an answer |
| `created_at`  | TIMESTAMP     | First seen                             |
| `last_seen_at`| TIMESTAMP     | Most recent occurrence                 |

The first table is metadata for document chunks; the others back the training
flow. The actual search is in Chroma.

## Configuration (`app/config.py`)

All knobs live in `.env`:

| Var                 | Purpose                                          |
| ------------------- | ------------------------------------------------ |
| `DATABASE_URL`      | SQLAlchemy DSN for Postgres                      |
| `CHROMA_PERSIST_DIR`| Where Chroma writes its files                    |
| `CHROMA_COLLECTION` | Collection name                                  |
| `CHROMA_WEB_COLLECTION` | Collection of allowlisted web passages       |
| `WEB_MAX_BYTES`     | Largest page body the reader accepts             |
| `WEB_TIMEOUT_SECONDS` | Time limit for one read or search              |
| `WEB_MAX_REDIRECTS` | Redirects a read may follow                      |
| `WEB_SEARCH_API_KEY` | Brave or Tavily key, read from the environment  |
| `SPACY_MODEL`       | spaCy model to load (sm / md / lg)               |
| `CORS_ORIGINS`      | Comma-separated list of allowed frontend origins |
| `BASE_ANSWER_CONFIDENCE` | The confidence a fresh answer starts with  |
| `MAX_ANSWER_CONFIDENCE`  | The highest confidence an answer can reach |
| `CONFIDENCE_STEP`   | How much a correction raises the confidence      |
| `CHROMA_INTENTS_COLLECTION` | Collection of taught intent wording   |
| `MAX_UPLOAD_BYTES`  | Largest uploaded file the request reads (default 25 MB) |
| `INGEST_WORKERS`    | Uploads one process reads at once; the rest wait in the queue |
| `PII_SCRUB_ENABLED` | When on, emails, phone numbers and long identifiers are masked before indexing |
| `WEIGHTS_CACHE_BYTES` | Memory the loaded model weight files may occupy before the least recently used model is let go (default 1.5 GiB) |
| `RERANK_ENABLED` / `RERANK_MODEL` | Optional cross-encoder pass over the fused passages, off by default |

`settings.cors_list` splits the CSV into a list for the CORS middleware.

## Extending the system

Four of the six next steps this list used to carry are now the system itself: an
answer model writes and streams answers, the `users` table and the single
workspace own every row, `GET /documents` and `DELETE /documents/{source}`
inspect and reset the training data, and uploads are read on a bounded pool.
What is genuinely still ahead:

1. **A new file type.** Add the extension to `core.files.ALLOWED_UPLOAD_EXTENSIONS`, teach `sniff_extension` its magic bytes and give `core.extraction` a reader that returns pages with their character spans; the queue, the quality pass, the collapse and the vectors then work unchanged.
2. **Better embeddings.** `services/embeddings.py` asks ONNX Runtime for the CPU-only MiniLM. Another model means re-embedding every collection with `scripts/rebuild_index.py` and re-measuring retrieval before it serves.
3. **A reranker.** `RERANK_ENABLED` and `RERANK_MODEL` are read by the request path and no cross-encoder is installed; turning one on is a measurement task, not a code task.
4. **Live web evidence measured.** The web layer is offline-tested only; a real SearXNG or search key and its latency are still missing from `docs/07-model-measurements.md`.
5. **Streaming answers from a remote model.** The local stream is in place; a remote provider would need the same stage events and the same refusal rules.

## Conventions

- **Code style:** PEP 8, 4-space indent, type hints on public functions.
- **Imports:** direct submodule imports (`from app.services.quality.score import passage_quality`, `from app.services.nlp import chunk_text`). A package `__init__.py` carries its docstring and, where the package is small, its re-exports; nothing imports across services through `app.main`.
- **No comments** unless something is non-obvious; this is enforced by the project's code-style rule.
- **No commits** to git unless explicitly requested.