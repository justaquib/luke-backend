# The Model's Measurements — Real Numbers Only

> One place for what Luke's tiny model actually measures, kept current after
> every improvement. Every number here was measured on this machine against
> the material the app really trains on. Nothing estimated, extrapolated or
> "would probably be" — the strict rule in `../../CODE_STANDARDS.md` forbids
> that. How the frontier labs measure the same qualities:
> `docs/06-frontier-model-evaluation.md`.
>
> **Retired on 2026-10-25, partially restored on 2026-10-26:** the from-scratch
> answer writer and its trace were removed from the app; their rows are kept
> exactly as measured, marked as history. The measured reason: on the held-out
> probes the writer was served **0** times — every cross-form and cross-topic
> answer came from the taught material, and the writer refused **0 of 8**
> adversarial probes on its own. The Hugging Face base-model teacher was
> restored on 2026-10-26 as a model the chat can ask directly (it writes from
> the retrieved reference, measured attention included), not as a source of
> training text for Luke. The word model is still Luke's only model.

## The measuring drill

Every row below says which kind of read it is, and the kinds are not
interchangeable:

1. Build the training set exactly as production does (`training_materials.collect`,
   `quality.keep`, `examples.word_example`), with the same vocabulary the model
   actually holds.
2. Split it into a train set and a held-out set by group — the group is the
   normalized response, so the same answer never sits on both sides
   (`measurement.group_split`).
3. Read the model in its resting state (`eval` mode, dropout off, `torch.no_grad`)
   over fixed batches built once for the run (`measurement.fixed_batches`) —
   the same read `training._check` does for the panel. A read that takes fresh
   random batches is a *training-batch read*, not a held-out measurement.
4. Report: the train read and the held-out read separately, the number of
   examples behind each, the number of passes, the step the best held-out read
   happened at, and why the run stopped.

Chance level for orientation: `ln(vocabulary)` is what pure guessing scores.
A loss of L means the model, at each word, puts about `e^-L` of its chance on
the exactly right token, or put the other way, hesitates between roughly `e^L`
equally likely words.

Every row also says whether the measurement used the corpus of that day. The
corpus grows as the trainer adds lessons, so a held-out number belongs to its
corpus and is never compared across corpora as if it were the same test.

## The answer model (retired 2026-10-25)

The rows below measured the from-scratch answer writer before it was retired.
They are kept as the record of what it did, not as the current model.

| When | Model state | Train read | Held-out read | Chance | Best step / steps | Vocabulary | Examples (train / held-out) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23 | Saved writer before the fix — `loss` was one random batch of 16; 2,700 steps ≈ 1.2 passes | **2.84** (training-batch read) | not measured; the panel number was noise (`2.4127`) | `ln 2092 = 7.65` | — / 2,700 | 2,092 | 36,374 |
| 2026-10-23 | Old flat loop — 3,000 steps, batch 16 | **~6.8** (training-batch read) | not measured | `ln 1644 = 7.40` | — / 3,000 | 1,644 | 36,374 |
| 2026-10-23 | Old loop — 9,000 steps ≈ 4 passes; still falling at the end | **1.377** (training-batch read) | not measured | `ln 1644 = 7.40` | — / 9,000 | 1,644 | 36,374 |
| 2026-10-23 | Saved writer on disk, `transformer-0.pt`, stopped at 1,700 of 9,292 planned steps *(superseded; that file has since been retrained)* | **7.3708** (training-batch read) | not measured | `ln 2187 = 7.6903` | — / 1,700 | 2,187 | 36,619 |
| 2026-10-24 | **Live checkpoint** `transformer-0.pt` — early-stopped; the loop that wrote it reported `passes: 0` because it did not count passes | **5.2055** (fixed-batch read) | **5.4964** | `ln 2292 = 7.7372` | 100 / 900 | 2,292 | 33,675 / 3,772 |
| 2026-10-24 | **Production default** — one pass, lr 5e-4, seed 7, early stopped, 36.0 s | **4.3342** (fixed-batch read) | **4.9207** | `ln 1844 = 7.5197` | 400 / 1,200 | 1,844 | 33,109 / 3,686 |
| 2026-10-24 | **Production default after conversation material** — one pass, lr 5e-4, seed 7, 2,113 steps, 210.2 s; corpus 17,672 contexts / 17,872 instructions, vocabulary 1,872 *(drill only, nothing saved)* | **2.0149** (fixed-batch read) | **3.7497** | `ln 1872 = 7.5348` | 2,100 / 2,113 | 1,872 | 33,799 / 3,761 |
| 2026-10-24 | Learning-rate matrix, old loop — lr 5e-4, 2,000 steps (corpus of that day, vocabulary 1,644, so not the production row) | — | **3.8951** | `ln 1644 = 7.40` | — / 2,000 | 1,644 | 36,374 |
| 2026-10-24 | Learning-rate matrix, old loop — lr 3e-4, 2,000 steps | — | **3.9334** | `ln 1644 = 7.40` | — / 2,000 | 1,644 | 36,374 |
| 2026-10-24 | Learning-rate matrix, old loop — lr 1e-3, 2,000 steps | — | **4.90–5.17** (bounced) | `ln 1644 = 7.40` | — / 2,000 | 1,644 | 36,374 |
| 2026-10-24 | Learning-rate matrix, old loop — lr 3e-3, 2,000 steps | — | **5.67** (peaked early, then worsened) | `ln 1644 = 7.40` | — / 2,000 | 1,644 | 36,374 |
| 2026-10-24 | Learning-rate matrix, old loop — 8 passes, 16,560-step horizon | — | **4.924** (the schedule never annealed) | `ln 1644 = 7.40` | — / 16,560 | 1,644 | 36,374 |
| 2026-10-24 | Smoke run of the current loop — 300 steps, batch 16, seed 7, nothing saved | **4.371** (fixed-batch read) | not measured | `ln 1742 = 7.4628` | — / 300 | 1,742 | 36,619 |

What the rows say in one line each:

- The panel used to show **2.4127** while the saved model truly read at
  **2.84**: the panel number was one unlucky batch, not the model.
- The old flat 3,000-step training was materially too short: at step 3,000 the
  model hesitated among `e^6.8` ≈ 900 words — barely better than guessing.
- More passes alone did not fix it: the 16,560-step schedule never annealed
  and stopped at **4.924**. An annealed one-pass schedule reaches a lower
  held-out loss in a fraction of the time.
- The live checkpoint reads **5.4964** held-out against its own chance of
  **7.7372** — it knows something, but it is not the production run: it
  stopped at step 900 because its held-out read was lowest at step 100, then
  rose for every check after.
- The production run reads **4.9207** held-out against **7.5197** chance, with
  its best read at step 400 of 1,200. The train read keeps falling after that
  (4.3342) while the held-out read rises: on 33,109 examples a 2.18M-parameter
  model begins to fit the training set, and early stopping is what keeps the
  reported model at the best held-out state.
- The learning-rate matrix chose the rate production uses: 3e-3 diverges,
  1e-3 bounces between 4.90 and 5.17, and 5e-4 and 3e-4 are the two best. It
  is a different corpus from the production row, so 3.8951 is evidence for the
  *rate*, not a lower bound for the product.
- The conversation material grew the corpus to 1,872 words. The production
  drill reads **3.7497** held-out against **7.5348** chance, with the train read
  at **2.0149**, train accuracy 0.673 and held-out accuracy 0.399. It reached
  the last of its 2,113 steps (one pass) without early stopping. This is the
  row for the 2026-10-24 corpus, which now includes the greetings and the
  model's own name, so it is not a like-for-like comparison with the 4.9207
  row above.

## The answer trace (retired 2026-10-25)

The answer details panel used to read one question through the saved writer and
show what the model really did (`POST /language-model/trace`,
`app/services/neural/tracing.py`). Both the endpoint and the module are gone;
the measurement below is kept as history.

Measured on 2026-10-24 against tenant None's saved checkpoint
`transformer-0.pt`, question "who is your creator?":

| What | Measured |
| --- | --- |
| Model | 4 layers, 8 heads, width 192, context 256, vocabulary 2,292, 2,709,120 parameters, CPU |
| Tokens read | 8: `<start> <instr> who is your creator ? <resp>` — 4 words |
| Embeddings shown | first 8 of 192 dims per token, with the full-vector norm |
| Attention | 4 layers × 8 heads × 8 × 8; every row sums to 1 (0.999998–1.000002 as stored), and the causal mask is exact — the first token gives 0 weight to every later token |
| Next-word candidates | `we` 0.16744, `i` 0.05182, `a` 0.02516, `my` 0.02013, `an` 0.01585 |
| Writer's run | `we have one have with we.` — 8 choices, stopped on the end mark in 0.018 s |
| Response payload | 20,272 bytes |

What these numbers say: the writer's next choice after the question was "we"
with only 0.167 of its probability, and its eight chosen words did not answer
the question. The trace reported that as it was — the real candidates and the
real run, never a cleaned-up example. The served answer for this question came
from another route (conversation, taught material or retrieval), and the panel
labelled the trace as the writer's measured reading of the same question
rather than the path that answer took.

## The word model

The word model is the served model of the neural engine. It is measured with
the same drill and its own chance level.

| When | State | Held-out loss | Train read | Chance | Next-word accuracy / chance | Perplexity / chance | Steps | Params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-24 | 8 passes, seed 7, serving gate passed | **3.1327** | **2.6567** | `ln 1523 = 7.3284` | **0.4571 / 0.0007** | **22.94 / 1,511** | 928 | 599,936 |
| 2026-10-24 | 8 passes, seed 7, after conversation material, serving gate passed | **3.0976** | **2.674** | `ln 1563 = 7.3544` | **0.4792 / 0.0006** | **22.14 / 1,563** | 944 | 606,592 |
| 2026-10-26 | 8 passes, seed 7, after the reading work; corpus 2,421 unique sentences (2,179 train / 242 held out), serving gate passed | **3.1008** | **2.667** | `ln 1563 = 7.3601` | **0.4694 / 0.0006** | **22.22 / 1,563** | 944 (best at 900) | 606,208 |

The serving gate (`serving.py`) accepts this model: its held-out loss 3.1327
is far below chance 7.3284, and its next-word accuracy is 457 times the
chance of picking the right word from 1,511. The 3.1327 row was measured over
206 held-out examples / 2,772 scored positions from 2,055 sentences
(29,968 transitions).

The conversation material also grew the word corpus: 2,097 sentences,
30,746 transitions and a 1,575-word model vocabulary. The gate still accepts
the model (held-out **3.0976** against **7.3544** chance, next-word accuracy
**0.4792** against **0.0006**). The drill's own per-position read is a
different read from the fixed training batches: 210 held-out examples /
2,790 scored positions score 0.3165 top-1 accuracy and 55.96 perplexity, and
both reads are kept here so they are not confused. The corpus changed, so
this row belongs to its own corpus and does not replace the row above.

The 2026-10-26 row re-measured the unchanged word-model code after the
reading work, on the same command (`scripts/measure_words.py --tenant 0
--seed 7`). The script collected 2,421 unique sentences; the task's grouped
split kept 2,097 sentences / 30,746 transitions, 1,887 train and 210
held-out examples, and a 1,572-word vocabulary. The run made 944 steps (best
at 900) in 30.7 s on MPS, did not stop early, and the serving gate accepted
it: held-out **3.1008** against **7.3601** chance, next-word accuracy
**0.4694** against **0.0006**. The drill's per-position read scores 2,790
positions at 0.3247 top-1 accuracy and 55.42 perplexity. The code did not
change; the corpus did, so this row belongs to its own corpus.

## The passages the training keeps (measured 2026-10-27)

The training bridge cuts each stored chunk into 3–50-word passages and keeps
them best quality first inside a 100,000-word budget
(`app/services/language/document_passages.py`). Read against tenant 0's real
material with `training_materials.gather(0, db)` and `document_passages(...)`
on this machine; nothing was trained or saved.

| The passage budget | Measured |
| --- | --- |
| Document chunks | **387** |
| Chunks kept / dropped | **387 / 0** |
| Passages kept | **1,314** |
| Passages dropped at the budget | **0** |
| Words in the kept passages | **53,891** of a **100,000** budget |

This corpus uses just over half its passage budget, so nothing is dropped and
the ordering by quality has nothing to exclude here. The material the bridge
reads in total is **1,697 lines**: lesson 500, starter lesson 533, conversation
29, approved answer 24, correction 210, document 387, word 1, sentence 13.

## Reading, connecting and finishing (2026-10-25)

Measured on tenant 0's real material (1,032 lesson cards) with
`.venv/bin/python scripts/measure_reading.py --tenant 0`. The script prints
one JSON object and never trains or saves a checkpoint.

### Reading

| What | Measured |
| --- | --- |
| Questions read | 12 |
| Words read | 63 |
| Words with a dictionary meaning | 24 |
| Words with synonyms | 21 |
| Question types | lookup 8, definition 3, counting 1 |

The 12 questions are fixed in the script. The types are the reader's own
classification, not an answer key: "What is a circle?" reads as a definition,
"Why is the sky blue?" and the poem request as lookups, and "How many legs
does a dog have?" as counting. Only key words are looked up in the
dictionary, which is why 24 of 63 words carry a meaning: the rest are
function words and words the key-word pass did not select.

### The synonym fallback

Coverage first: the 1,032 lessons hold 777 distinct title words. 751 have a
WordNet entry; the other 26 have neither an entry nor a synonym. That
measurement changed the design. The first version ran the fallback only when
the dictionary had no entry, but the same WordNet gate that reports "no
entry" also returns no synonyms, so that path could never fire. The fallback
now runs on every definition question; a lesson reached through a single-word
synonym is served only when exactly one lesson is reached and it shares a key
word with the question, and otherwise the dictionary answer stands unchanged.

| What | Measured |
| --- | --- |
| Probes built from real lessons | 22 |
| Reached the intended lesson | **5** (22.73%) |
| Reached a different lesson | **0** |
| Reached no lesson | **17** |

Each probe asks "What is a {synonym}?", where the synonym is a WordNet synonym
of a lesson's title word and appears in that lesson's answer, the exact
lesson lookup does not already answer the question, and the dictionary holds
the question word. The five that passed: "What is a base?" →
"GK: What is a foundation?", "What is a fall?" → "GK: What is autumn?",
"What is a times?" → "What is multiplication?", "What is a pool?" →
"What is a pond?" and "What is a opposite?" → "What is an antonym?". The 17
misses are printed with their reasons; they are mostly synonyms outside the
fallback's six-synonym window. An earlier probe run, before the single-word
filter, reached one wrong lesson through the multi-word synonym "looking at";
phrases are now rejected and that misfire is gone. The fallback never guesses
between two lessons.

### Finishing

| What | Measured |
| --- | --- |
| Taught lesson answers finished | 1,032 |
| Already finished | 1,031 |
| Changed | **1** |
| Idempotent (`finish(finish(x)) == finish(x)`) | **1,032** |

The one changed answer is a story that ended without a full stop; finishing
added it. No answer lost a word, a capital or an existing mark.

## Training with a Hugging Face base-model teacher

The **Base model** tab trains a Hugging Face teacher on the training room's
material. It was retired on 2026-10-25 and restored on 2026-10-26 in a
different role: the chat can now answer with the saved teacher (it writes from
the retrieved lesson or FAQ and reports its attention), instead of the teacher
writing answers that then joined Luke's training material. The run below is the
2026-10-25 measurement and is kept as history for the fine-tune itself; it is
not a read of the restored serving path.

| The teacher | Measured |
| --- | --- |
| Model / parameters / device | SmolLM2-135M, 134,515,008, MPS |
| Fine-tune steps / examples | 200 / 400 |
| Fine-tune loss (last step) | **0.75** |
| Mean of the last 20 steps | **1.023** |
| Fine-tune time | 126.3 s |
| Teacher answers kept | **19 of 24** questions; 5 dropped by the 3–48-word / no-loop guard |
| Saved teacher | **272,585,201 bytes** under `huggingface/tenant-0/teacher-HuggingFaceTB-SmolLM2-135M/` |

| Luke after the lesson | Measured |
| --- | --- |
| Writer train / held-out loss | **4.2894 / 4.9552** |
| Writer steps / time | 900 / 63.1 s, stopped early |
| Word model train / held-out loss | **0.8578 / 1.6749** |
| Word model steps / time | 4,200 / 129.9 s, stopped early |
| Intent / emotion examples | 1,062 / 66 |
| Material before the teacher wrote | 2,068 sentences and 17,872 requests |

The two tables answer different questions and are not on the same scale: the
teacher's loss is how well it fitted the 400 demonstrated answers during its
fine-tune, while Luke's losses are how well he fits the enlarged material,
read on a held-out split. The page keeps them apart for that reason.

Two honesty notes belong with these rows. First, a teacher answer that
survives the guard is used exactly as written: the run does not fact-check it,
and a measured answer from this run was factually wrong. Second, the grouped
validation split shifts when the teacher's answers join the corpus, so these
Luke losses belong to this corpus and are not a like-for-like continuation of
the rows above. The numbers are one honest measurement of one run, not an
average; early stopping and MPS make another run possible to differ. The page
prints "not measured" for any value a run did not measure.

The default of 200 fine-tune steps is itself a measurement: a 60-step run
repeated its own prompt in every answer, so the page starts at 200 (the range
is 1 to 2000).

## Serving the restored teacher (measured 2026-10-26)

The restored chat path was read against the installed
`teacher-HuggingFaceTB-SmolLM2-135M` on the same machine (Apple silicon, MPS,
`torch` 2.14.1, `transformers` 5.20.2). The teacher has **30 layers, 9 heads,
hidden size 576**.

| The teacher's read | Measured |
| --- | --- |
| First prediction (weights load + forward) | **6.13 s** |
| Warm answer | **2.52 s** |
| Attention returned for `The sky appears blue because` | **5 × 5**, last layer averaged over heads, peak **1.0**, row sums 0.999–1.002 |
| Next words | `the` **23.14%**, `of` **20.41%**, `it` **18.07%** |
| Wording accepted of 6 probe references | **1 of 6** (`Paris is the capital of France`) |

The 1-of-6 row is the important one. The teacher frequently continues with
facts of its own rather than rewriting the reference — for example, asked why
the sky is blue with the reference above, it wrote *"…This makes it easier for
us to see in the dark."* The material guard (`_uses_material`) accepts an
answer only when at least two of its content words come from the reference and
make up a third of it, so that answer and four others were refused and the
matched taught answer was served instead; the `model` connection records the
fallback. The guard is a lexical measure and is not a fact check: an accepted
rewrite can still be wrong. The page never presents a teacher answer as Luke's.

## Moving teacher knowledge into Luke (measured 2026-10-26)

The transfer plan was read against tenant 0 on this machine with the installed
teacher present, before any run was started:

| The plan | Measured |
| --- | --- |
| Questions the material can still ask | **1,050** |
| Lessons / starter lessons / corrections | **500 / 533 / 4** |
| Taught words | **1** |
| Topic questions (subjects, capped) | **12** (of 12 subjects) |
| Document questions | **0** for this tenant |
| Default run size | **40**, clamped to 1–200 |

The same answer guard is used here, against the reference the teacher was
given. In the transfer test suite, of three written answers **2 grounded ones
were kept** as approved FAQs and indexed, and **1 ungrounded one was skipped**;
undo then removed exactly the two it had created and left the tenant's other
FAQs alone. These are test measurements of the guard and provenance rules, not
a claim about how often a particular teacher stays grounded.

## The downloadable package (measured 2026-10-26)

The preview and one real download were read against tenant 0 (the
administrator) on this machine, without starting a run:

| The package | Measured |
| --- | --- |
| File | `luke-0-2026-10-26.zip`, **2,539,103 bytes** |
| Material lines | **1,622** — lesson 500, starter lesson 533, conversation 29, approved answer 12, correction 147, document 387, word 1, sentence 13 |
| Unresolved questions kept for context | **12** |
| Checkpoints | **5** — word model 2,557,967 bytes; clarify 5,029; correct 5,029; emotion 14,309; intent 14,301 |
| Files in the zip | **13**, each listed with its size and SHA-256 |
| Teacher weights in the package | **0** — excluded by design |

The zip holds only the administrator's own rows and files (`tenant_id IS NULL`);
a tenant's package is built from that tenant's rows and files alone. That scoping
is proved in `tests/test_language_export.py`, not assumed.

## The package, format 2 (measured 2026-10-27)

`bundle.preview` and `bundle.build` were read against tenant 0 on this machine,
and the slim zip was then dry-run through `restore.import_package(0, db, ...,
dry_run=True)`. The dry run checks the schema, every file hash, the counts and
the local references and stops before touching a row, file or vector, so
nothing in the workspace moved for these rows.

| The package | Measured |
| --- | --- |
| Preview, full | **41** files, **1,019,363,363 bytes** |
| Preview, slim | **32** files, **19,764,148 bytes** |
| Full zip | `luke-0-2026-10-27.zip`, **1,009,100,162 bytes**, built in **2.11 s** |
| Slim zip | `luke-0-2026-10-27-slim.zip`, **17,230,663 bytes**, built in **0.40 s** |
| Base checkpoint in the full package | **9** files, **999,599,334 bytes** — Luke 0.0.1, Apache-2.0, weight SHA-256 recorded |
| Dry run of the slim zip | **0.71 s**; **32** files / **19,764,148 bytes** checked; `replace_required: true` against this non-empty workspace |
| Material lines in the manifest | **1,697**, as in the passage section above |
| Rows in the manifest | documents 387, FAQs 24, unresolved pending 16 of 21, feedback 35, uploads 0, intents 0 |
| Derived counts | 1,314 passages, 5 checkpoints, 4 tokenizers, 4 raw files |

The full zip is almost entirely the base checkpoint: 999.6 MB of the 1,009.1 MB
zip is `weights/base/`, and the workspace's own material is the 17.2 MB the
slim zip carries. Both modes are built from the same state, so the difference
between them is the base weights and nothing else.

The first real dry run failed with `A Feedback row is missing question.`:
**11 of the workspace's 35 ratings** were recorded against an answer whose
question the app did not keep, so their `question` column is empty. The app
stores those rows, so the package must carry them; the restore now writes them
back with an empty question instead of refusing the whole package, and
`tests/test_package_import.py::test_feedback_without_a_question_is_exported_and_restored`
covers both the empty value and a line written without the column at all.

## Rebuilding the index in a cosine space (measured 2026-10-27)

The collections were rebuilt from Postgres with cosine space, HNSW settings
M 16 / construction 200 / search 100 and UUID5 ids, verified against the
database, then the old collections were deleted and the store vacuumed. The
numbers are what `backups/index-rebuild-20261027-230714/report.json` recorded
during the run.

| The rebuild | Measured |
| --- | --- |
| Collections before | 387 documents, 24 FAQs, 16 intents, all in Chroma's default `l2` space |
| Store before → after | 20 files / **287,171,593 bytes** → 32 files / **244,206,773 bytes** |
| Bytes saved by the rebuild | **42,964,820** |
| Vacuum | **284,278,784 → 232,390,656 bytes**, **51,888,128 bytes** saved |
| Collections after | 387 / 24 / 16, `hnsw:space` cosine, M 16, construction 200, search 100 |
| Verification before the swap | FAQs and intents found their sample at rank 1; the document sample — a chunk's first 12 words, not a question — did not return in the top 5 |
| Verification after the swap | all three collections returned their sample at rank 1 |
| Same query's similarity | **0.8** read through the old `l2` label, **0.6** in the new cosine space |

The old store was smaller in file count but larger in bytes because Chroma's
`l2` segments had not been compacted; the rebuild wrote a smaller store and the
vacuum compacted it again. The two savings are measured against different
snapshots (the rebuild against the pre-rebuild directory, the vacuum against
the store as the run found it), so they are reported separately and not added
into one total. The document sample is the one check that read `ok: false`
before the swap: its query was the first words of the chunk ("By T . Albert
Illustrated by…"), which is a caption rather than a question, and the check was
rerun against the canonical collections with the same sample, which returned
at rank 1.

## Reciprocal rank fusion (measured 2026-10-27)

The app now fuses the semantic and lexical rankings by reciprocal rank
(`fuse_rankings`, k = 60) instead of the weighted sum the earlier comparison
used. Re-run against the same 24 question/answer pairs and the same in-memory
embeddings, on the rebuilt index:
`.venv/bin/python scripts/measure_retrieval.py --source 0/React_JavaScript_NextJS_Interview_QA_Guide.pdf`.

| Measurement | Semantic only | Fused by reciprocal rank |
| --- | --- | --- |
| Top-1 | 22 of 24 | **23 of 24** |
| Top-4 | 23 of 24 | **23 of 24** |
| Chunks / open boundaries | 47 / **0** | 47 / **0** |
| Answers wholly inside one chunk | **24 of 24** | **24 of 24** |

The old splitter's chunks are no longer in the index after the rebuild, so this
run reports **0** old chunks and the old/new columns of the earlier table
cannot be reproduced; those rows belong to their own index and are kept as
history. Top-4 fused reads 23 of 24 here where the earlier weighted run read
24 of 24 — a different fusion rule on a rebuilt index, so the difference is
reported rather than smoothed over.

## The FAQ and intent thresholds (measured 2026-10-27)

`scripts/calibrate_thresholds.py` reads every rating the workspace holds
against its current FAQs and prints the distances the labelled rows support.
It only chooses a threshold when at least 3 positives and 3 negatives exist,
and says so when they do not.

| The thresholds | Measured |
| --- | --- |
| FAQs / feedback rows read | **24 / 35** |
| Current FAQ match threshold | **0.28** |
| Current intent match threshold | **0.36** |
| FAQ positives / negatives | **1 / 17** |
| Distance the FAQ rows suggest | **0.223** (accuracy 1.0, margin 0.4455) |
| Intent rows that support a threshold | **0** disliked answers |
| Thresholds changed | **none** — the script reported `enough: false` for both |

The single positive is closer than every one of the 17 negatives, the nearest
of which sits at 0.4455, and the midpoint the script would choose is 0.223.
One labelled example is not a measurement of a threshold, so the value stays at
the 0.28 production already uses and the midpoint is reported only as what the
evidence holds. The intent threshold has no disliked answer to read at all.


## Retrieval: chunking and hybrid search

Measured on the uploaded `React_JavaScript_NextJS_Interview_QA_Guide.pdf`,
24 question/answer pairs, every pair with an answer of at least 24 words.
Old = the chunks stored in Chroma by the previous splitter; new = the current
sentence-aligned, question-aware chunker. Both were embedded with the same
MiniLM model and reranked by the same hybrid rules.

| Measurement | Old chunks | New chunks |
| --- | --- | --- |
| Chunks for the document | 17 | 47 |
| Boundaries that end mid-sentence | **12 of 16** | **0 of 46** |
| Answers whose text is wholly in one chunk | 21 of 24 | **24 of 24** |
| Top-1, semantic only | 20 of 24 | **22 of 24** |
| Top-1, hybrid (0.4 semantic + 0.6 lexical) | 20 of 24 | **23 of 24** |
| Top-4, semantic only | 20 of 24 | **23 of 24** |
| Top-4, hybrid | 21 of 24 | **24 of 24** |

The open boundaries are the reason the new chunker exists: twelve of the old
sixteen chunk edges cut an answer in half, so no embedding of that chunk could
hold the whole answer. The new chunker starts a question on its own chunk and
only overlaps whole sentences, which is why the answerable set goes from 21 to
24 and top-4 hybrid reaches 24 of 24.

## The similarity label (measured 2026-10-27)

Read from the persisted Chroma collections: `docs` 387 vectors, `faq` 12,
`intents` 16, each with no `hnsw:space` metadata, which is Chroma's `l2`
default. A squared L2 distance of `d` is now reported as similarity `1 - d/2`;
a cosine space would be `1 - d`. Before this change every distance was read as
cosine, which reported a lower similarity than the stored distance supports.

No threshold was moved with the label. The numbers the chat shows and the
thresholds that gate an answer stay separate, and recalibrating them belongs to
the Phase 2 index rebuild where the collections are rebuilt with a declared
space.

## The foundation checkpoint — Luke 0.0.1 (measured 2026-10-27)

`scripts/measure_foundation.py` loads the installed checkpoint through the same
loader the app uses and reads every number below from that run. The checkpoint
is pinned, read-only and verified against a recorded hash; nothing in this
section was written by Luke, and no weight was changed.

| Item | Value |
| --- | --- |
| Model | Luke 0.0.1 |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Licence | Apache-2.0 |
| Parameters | 494,032,768 |
| Context limit | 32,768 tokens |
| Weights | BF16, 988,097,824 bytes (942.3 MB) |
| Weight SHA-256 | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| Device | mps (float16); CPU fallback is float32 |
| Cold load | 7.64 s |
| One fixed prompt | 75 new tokens in 2.986 s = **25.117 tokens/second**, stopped on EOS |
| Peak resident memory | 1,931,182,080 bytes (1,841.7 MB), after load and two answers |
| Held-out perplexity | **7.272** on a 99-token general-text sample (water cycle) that is not taught |
| Streaming | first piece after 0.086 s, 35 pieces, 1.386 s total, greedy; the prompt is not echoed |

The prompt was "Explain in three sentences why the sky looks blue during the
day." and the answer was three grammatical sentences about scattering. The
streaming check asked for a colour in one short sentence and received one.
Speed is measured on this machine (Apple silicon, MPS) and will differ
elsewhere; memory is the process peak, not the checkpoint's file size.

The checkpoint is a writer of last resort, not a source of knowledge. A taught
answer always wins; when nothing is taught the answer is labelled written and
unverified, carries no citation, and is never stored as a lesson, FAQ or
training pair. That boundary is what keeps Luke 0.0.1 lightweight: the base
model adds wording, not garbage knowledge.

## What the model does with what it knows

Measured 2026-10-24 on the live tenant through the writer's instruction test
(then stored in `instruction-0.json`; the test was retired with the writer on
2026-10-25). The rows are kept as the record that led to the retirement.

| Measurement | Result | What it means |
| --- | --- | --- |
| Never-worded requests (cross-form) | **12 of 12**, written **0**, looked up **12** | the request is matched to the taught lesson; the writer's own wording did not pass the grounding and loop checks, so the taught answer is served |
| Held-out subject (cross-topic) | **6 of 6**, written **0**, looked up **6** | a subject kept out of the probe training is answered from the taught material |
| Adversarial refusals (product) | **8 of 8** | an unknown topic is never answered: the material boundary returns no answer and the user gets the "not answered yet" reply |
| Adversarial refusals (writer alone) | **0 of 8** | the 2.2M-parameter writer does not learn to refuse by itself; the boundary is the guard, and both numbers are shown so this is not hidden |
| Probe contamination | **0 of 12** held-out forms in the training | the probes measure what they claim to measure |

The cross-form and cross-topic numbers are perfect *because* the taught answer
wins every time. The `written` column is the honest part: the writer was
served **0 of 12** forms and **0 of 6** subjects. That is the measured reason
it was retired on 2026-10-25 — the taught answer is the correct path, not a
fallback around a model that works.

The two "Describe … to me" probes are refused *because the material boundary
finds nothing*, not because the request is unreadable: `shapes.py` reads that
wording as a fact request, so the same wording about a taught topic ("Describe
the wind to me.") finds the lesson and is answered. Without that shape the
probes would have been refused before the topic was even looked up, and the
score would have meant nothing. The four negation probes are refused one step
earlier: a poem that is *not* about a taught topic is not a request the
product can satisfy.

## Changes and what they did

Rows that mention the writer, the trace, the scaffold or the base-model teacher
describe changes made while those features existed; they are kept as history.
The reader, fallback and finishing changes of 2026-10-25 are measured in their
own rows below the table, and in the section that follows it.

| Change | Before | After |
| --- | --- | --- |
| Loop guard (`quality.repeats_itself`, `MAX_PAIR_SAYS = 2`) | "write a poem on rain" → `rain, go away, go away, go away, go away` (grounded, so given away) | the writer's loop is refused; the full taught poem is given instead |
| Kind-scoped material search (`requests.find_material`) | "write a poem on star" refused even with the poem taught — "star" is said by 8 fact lessons, so the topic index dropped the word | the kind the request asked for picks the pool; the poem is matched and answered |
| Poem requests in the words parents use (`shapes.py`, `requests.py`, `answering.py`, `parenting/chat_engine.py`) | "tell me a poem on star" was not read as a request and "stars" did not match "star"; the coach wrote from an unrelated card and showed a "Needs a better answer" scaffold | "tell me/read me/recite/sing me" are read as requests, plural topics match their singular lesson, and a request for a poem is answered before a lesson *about* poems; the taught poem is served with no scaffold. The instruction probes re-measured unchanged: 12 of 12 cross-form, written 0, refusals 8 of 8, cross-topic 6 of 6 |
| Lesson answers and the coach card (`matching.py`, `parenting/chat_engine.py`) | "We see through which body part?" shared no words with the eye lesson's title, so no lesson matched; the coach then used the first card of its session topic ("General") — the Shakespeare card — and reported it as taught with confidence 1 | a lesson's answer can place it when its title cannot: the eye lesson is found through "The eye is a body part. It helps us see.", a title match still wins, and at least two shared words are required. The coach picks the card the message covers, never one that only shares the topic. The instruction probes re-measured unchanged: 12 of 12 cross-form, written 0, refusals 8 of 8, cross-topic 6 of 6 |
| Truthful checkpoints + one-pass training (`training._check`, `TRAINING_PASSES`) | panel showed a single batch's loss; early stopping decided on that noise; the training saw the material ~1.2 times | panel shows the averaged resting-state read; stopping decides on that; production makes one annealed pass (4.9207 held-out) |
| Stored instruction measurement (`brain.remeasure_instruction`, `instruction-{tenant}.json`) | the instruction test re-ran on every brain read — a brain page load took ~59 s | a brain page load reads the stored measurement in ~1.5 s; a fresh one is taken when the trainer presses the test button or a training finishes |
| Level ladder in hundredths with an approval gate (`brain.LUKE_LEVELS`, `approve_level`) | the level auto-advanced to whatever the measurements said and borrowed model-level names for the volume scale | the ladder is 0.0.0 → 0.0.9 → 0.1.0; the measured and displayed levels are separate and the trainer approves a promotion |
| Sentence-aligned, question-aware chunking (`nlp.chunk_text`) | 12 of 16 chunk edges cut a sentence; 21 of 24 answers fit in one chunk | 0 of 46 open edges; 24 of 24 answers fit; top-1 hybrid 23 of 24 |
| Hybrid reranking (`retrieval.rerank_hits`) | embedding distance alone | 0.4 semantic + 0.6 lexical, stopwords neutral: top-1 20 → 23 of 24, top-4 21 → 24 of 24 |
| Held-out refusal probes (`negatives.build_held_out_negatives`) | refusals were scored on the same requests the training showed | the probes are requests the training never showed; the product refuses 8 of 8 and the writer's 0 of 8 is reported beside it |
| Conversation material (`language/conversation.py`, `training_materials`) | a greeting, the model's name, thanks and a goodbye only reached the answer path through the taught material; the name was fixed | the pairs are answered directly and trained, and are built from the model name in Settings; the production drill on the new corpus reads 3.7497 held-out against 7.5348 chance |
| Base-model teacher (`huggingface/service.py`, `tuning.py`, `serving.py`) | the 2026-10-25 teacher only wrote answers that then trained Luke; it was never served, and the 2026-10-26 restored teacher was first guarded by a one-word overlap | the teacher is now answerable from the chat and grounded on the retrieved reference: of 6 probe references **1 of 6** wordings passed the stricter guard (two shared content words and a third of the answer), and the other 5 fell back to the matched answer. Attention and next words are real (30 layers / 9 heads, row sums 1.0) |
| Reading before answering (`nlu/`, `answer_flow.py`, `answer_streams.py`) | the question went straight to retrieval and one typing bubble hid every step until the whole answer arrived | the question is read (12 probe questions: 63 words, 24 meanings, 21 synonym sets), then each source reports what it found or did not, and the JSON path and the stream consume the same events; nothing shown is invented by the page |
| Synonym-aware lesson fallback (`language/fallback.py`) | a definition question that named no lesson title went to the dictionary even when a taught lesson answered the same idea in other words | WordNet synonyms of the question word are tried against the allowed lessons; a probe set of 22 real lessons reached 5 (22.73%) and picked a **different** lesson in 0; the rest stay on the dictionary. The approved plan scoped this to questions with *no* dictionary entry, but that state cannot occur: the entry lookup and the synonym expansion share the same WordNet gate, so a question without an entry has no synonyms either (777 lesson-title words measured: 751 entries, 26 with no entry and no synonyms). The fallback therefore runs on every definition question and overrides the dictionary only when exactly one lesson is reached and it shares a stemmed key word with the question |
| Answer finishing (`language/finishing.py`) | a lowercase taught answer with no full stop was served exactly as stored | outer whitespace trimmed, first alphabetic character capitalized, and a full stop added only when the text ends in a letter or digit; no word is changed. Across 1,032 stored answers 1,031 already read as sentences, 1 changed (it gained its full stop), and all 1,032 are idempotent |
| Foundation checkpoint (`foundation/`, chat generation) | the chat served only taught material; the 606K-parameter word model was the only writer and was not competent | the pinned Qwen2.5-0.5B-Instruct checkpoint writes only when no taught answer exists: 75 tokens in 2.986 s (25.117 tok/s), 7.272 perplexity on held-out prose, 1,931,182,080 bytes peak. The checkpoint is read-only, its answer is labelled written and unverified, carries no citation, and is never stored as knowledge — taught answers still win |
| Conversation expressions (`language/expressions.py`, `/dashboard/expressions`) | opening and closing lines were fixed in code and only four built-in small-talk pairs reached the answer path | the trainer teaches any wording under open/close/pause/resume (whole-message match), a taught expression is answered before built-in small talk, and the pairs also join the training material |
| Scratch tokenizer (`neural/tokenizer/`, checkpoint schema 3) | one word-level vocabulary; a checkpoint recorded no tokenizer, and there was no subword path | a tokenizer belongs to the weights it trained with: the word adapter keeps every legacy id, byte-level BPE is available as an opt-in, artifacts are content-addressed and verified by hash, and a mismatch is refused. **Measured (same split, 600 steps): word 1.3298 bits/char at 620,416 parameters; BPE 1.4807 at 888,704** — the word tokenizer stays the default and BPE is not promoted |
| Cosine threshold calibration (`config.py`, `.env`, `scripts/calibrate_thresholds.py`) | the calibrated values in `config.py` were overridden by `.env`, so the old L2 limits `0.55`/`0.75` were applied to cosine distances. The nearest saved answer to "Write a short poem about the rain." was the Twinkle poem at **0.5339**, inside the old exact limit while the taught rain poem existed | `0.28` exact and `0.36` loose, both below the lowest measured negative (`0.4455`) and the calibration's best split (`0.223`, accuracy 1.0, 1 positive / 17 negatives). The rain regression (`tests/test_rain_regression.py`) reads **2 of 9 failed before, 9 of 9 passed after**; exact, paraphrase and partial-phrase wordings all serve the whole taught poem, and a question with no lesson is written and labelled `generated` |

## The scratch tokenizer experiment (2026-10-27)

`scripts/measure_tokenizer.py`, tenant 0, seed 7. The corpus is the material
the word model trains from: 2,861 sentences, 2,425 kept, split by whole groups
into 2,183 training and 242 held-out sentences (136,477 and 14,613
characters). A BPE vocabulary is built from the training side only; every model
row trained with the same 600 steps, width 128, 2 layers, 4 heads, context 64
and learning rate 5e-4. Bits per character = held-out nats per token × held-out
tokens per character ÷ ln 2, the only figure comparable across vocabularies.

| Tokenizer | Vocabulary | Held-out loss | Bits/char | Tokens/char | Accuracy | Parameters | Embedding parameters | Tokenizer artifact | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Word (baseline) | 1,683 | 3.8539 | **1.3298** | 0.2392 | 0.3761 | 620,416 | 215,424 | none | 2,529,035 B |
| BPE, 2,048 asked | 2,048 | 4.3209 | 1.7405 | 0.2792 | 0.3226 | 667,136 | 262,144 | 49,375 B | 2,735,371 B |
| BPE, 4,096 asked | 3,779 | 4.0979 | 1.4807 | 0.2505 | 0.3751 | 888,704 | 483,712 | 99,867 B | 3,665,291 B |
| BPE, 8,192 asked | 3,779 | 4.0979 | 1.4807 | 0.2505 | 0.3751 | 888,704 | 483,712 | 99,867 B | 3,665,291 B |
| BPE, 16,384 asked | 3,779 | 4.0979 | 1.4807 | 0.2505 | 0.3751 | 888,704 | 483,712 | 99,867 B | 3,665,291 B |

- Truncation at 64 tokens: word 0.0; BPE 2,048 **0.0041**, BPE 3,779 0.0. At
  128 and 256 both are 0.0, so context length is not where BPE could win here.
- Encode throughput: word 8,359,443 characters/s; BPE 2,164,902 (2,048) and
  1,883,341 (3,779) on the same 14,613 held-out characters.
- Held-out words a train-only word vocabulary could not hold: **17 of 3,474
  (0.49%)**, so the word model's full-corpus vocabulary does not flatter it
  materially against BPE's training-only vocabulary.
- The 136K-character corpus cannot fill a larger vocabulary: the 4,096, 8,192
  and 16,384 requests all produced the same **3,779-piece** vocabulary and the
  same artifact (`bff213f5…`, SHA-256), where the 2,048 request produced
  `0dd3e700…`.
- Training seconds in this run: baseline 22.8, BPE 2,048 39.0, BPE 3,779 46.4
  (the three identical 3,779 rows trained 27.5 and 23.4 s on re-runs).

**Decision.** The word tokenizer stays the default for new scratch training:
it is 0.15 bits per character better on held-out text and costs 268,288 fewer
parameters and a 1.14 MB smaller checkpoint. BPE is an explicit opt-in
(`tokenizer_kind: "bpe"`), remains fully tested and loadable, and every artifact
is content-addressed so a retrained tokenizer never silently replaces an older
one. No claim is made that BPE improves the model's intelligence; on this
corpus it does not.

**Verification (2026-10-27).** `./quality.sh` green: frontend 50 files and 225
tests, backend 473 tests, Ruff and the structure checker clean. The Brain
tokenizer card was read in a browser at 1440, 1024 and 390 px in four states —
a BPE checkpoint, a word checkpoint, no checkpoint, and a missing artifact —
with no console error and no sideways scroll; the missing state shows the
backend's error and a "Check again" button. A unit test proves the prediction
lab calls a BPE token a piece and joins it without adding a space.

## The frontier comparison — in kind, not in number

You cannot put Luke's 4.92 next to a frontier model's published loss and rank
them: the numbers sit on different corpora, different tokenizers, different
vocabularies. The honest comparison is the *discipline* of measurement, which
follows the same conventions (details in
`docs/06-frontier-model-evaluation.md`):

| The frontier convention | Ours (measured where shown) |
| --- | --- |
| Held-out, averaged, eval-mode loss — never a single batch | `measurement.fixed_batches`: fixed train and held-out batches built once per run, resting state — the word model's 3.0976 is held-out, 2.674 is its train read |
| Perplexity declared only against its own corpus/tokenizer | the word model's 22.14 against its own chance of 1,563 |
| Scaling laws: length of training measured against the material | eight passes over the examples, capped at 20,000 steps and 1,500 s; the word model's material is small, so one pass would stop at the minimum step count |
| Grounding: every sentence supported by the given material | the answer is served as the trainer wrote it, so nothing is written that needs grounding |
| Calibration: "I am not sure" must mean something | the loose-match flag marks a weak match instead of pretending it is sure |
| Refusal instead of invention about the unknown | the material boundary refuses 8 of 8 held-out probes |
| Contamination control: probes held out of training | the generality probes are held out of training, and the request probes are built apart (`negatives.build_held_out_negatives`) |
| Reporting failures beside successes | the brain page reports the held-out row, the run's stop reason and a measurement it could not take as "not measured" |

## Web evidence (Phase 4)

The web step was added on 2026-10-28. It changes the request path, so it is
recorded here even though most of what it does cannot be measured on this
machine yet.

| What | Measured |
| --- | --- |
| Live search latency, extraction speed, refresh duration | **Not measured yet.** No SearXNG instance and no Brave or Tavily key are configured on this machine, so there is no real search to time. They will be measured before a live number is shown anywhere. |
| Repeat question from a stored snapshot | **Not measured yet** as a latency figure; by design the stored path makes no network call, and the tests assert that a stored hit answers without one. |
| Refusals (SSRF, unapproved domain, redirect cap, size and type caps, robots) | **58 offline tests** in `tests/test_web_policy.py`, `test_web_fetch.py`, `test_web_store.py` and `test_web_routes.py`, using recorded fixtures and `httpx.MockTransport`; no test opens a socket to the internet. |
| Answer discipline | A web answer carries a citation, the URL, `as_of`, `verified: "false"`, and a connection that calls it unverified and not taught material. The verifier marks it understood but not grounded in taught material. |

What these rows say: the safety properties are tested offline and pass; the
speed and refresh numbers do not exist yet and are written as such rather than
invented.

## Ingest quality and dedupe (Phase 5)

The ingest path now measures every passage before it is stored. The change
alters which passages future training runs will prefer, so it is recorded here;
no model was retrained for it.

| What | Measured |
| --- | --- |
| Passage quality | `passage_quality = language_factor × text_density × (1 − repetition)`, rounded to four places. English scores at 1.0, an unconfirmed language at 0.5, fewer than three words or repetition at or above 0.6 scores 0.0. The tests assert a clean sentence scores 1.0, a repeated sentence 0.0, a fragment 0.0 and an unconfirmed language 0.5. |
| SimHash near-duplicate distance | 64-bit SimHash over word unigrams with Hamming threshold 5. Unit tests measure a one-word change at distance 4 (collapsed), a two-word change at 1 (collapsed) and unrelated chunk-sized text at 16 (kept). |
| SHA-256 exact duplicates | Collapsed within the same file and against every stored row of the workspace, at passage level. |
| Existing rows backfilled | 387 of 387 `documents` rows received a SimHash on 2026-10-28; the migration report is `backups/simhash-20261028-121229-521271/report.json`. |
| Existing rows scored | The same 387 rows had no language and no quality — they were written before the score existed, so a training run would have ranked the whole corpus below any new upload. `scripts/migrate_quality.py` measured them with the ingest path's own functions and filled **387 of 387** on 2026-10-28; report `backups/quality-20261028-155329-337930/report.json`, tests in `tests/test_migrate_quality.py` (3 passed, including "a second run fills nothing"). |
| Quality of the stored corpus | Of the 387 scored passages: **all 387 detected as English**; mean **0.9001**, median **0.9027**, p5 0.8146, p95 0.9817, min 0.0 (1 passage), max 1.0 (7 passages), **335 distinct** values. The `documents` table went from 816 kB to 856 kB for the two filled columns. |
| Effect on training | No model was retrained for this change. The score orders the candidate passages a run picks and filters nothing, and with every stored passage now scored the ordering is real for the whole corpus rather than only for new uploads. |
| Trainer surface | Every job state was driven in a real browser against the running backend at 1440, 1024 and 390 px, including the two error paths. It found a retry defect that doubled a source's passages; see "What the trainer sees while a file is being read". |

### How far apart two answers have to be to be a conflict

`SAME_ANSWER_SIMILARITY` is a word-set Jaccard: shared words over all words in
both answers, after casefolding and stripping punctuation. Measured against the
tests that fix the boundary in `tests/test_contradictions.py`:

| What | Measured |
| --- | --- |
| "Water falling from clouds." against "Water falls from clouds." | 3 shared / 5 union = **0.60** — under 0.8, so a conflict and it reaches the inbox |
| "Water falls from clouds." against "Water falls from the clouds." | 4 shared / 5 union = **0.80** — at the threshold, so a rephrasing and nothing is filed |
| Question matching | word-set equality: "what is rain" matches "What is rain?" (0 tests needed — the same four words in another order or punctuation) |
| What a conflict does | one inbox row with both answers in the note, `occurrences` counted up on a repeat instead of a second row; the approved answer is untouched and keeps serving |
| Coverage | **8 tests** in `tests/test_contradictions.py`, including another workspace answering into this inbox |

## Request-path speed (Phase 5)

The request path changed on 2026-10-28: the same passages and the same
ranking are used, but the work behind them is cached. Nothing here claims a
different answer — the tests assert the ranking still finds the named passage.

Measured on this machine with the live store (387 document chunks, 384
dimensions), Python 3.13, Chroma 0.5.5:

| What | Measured |
| --- | --- |
| HNSW settings actually in the live collections | `docs`, `faq`, `intents`, `web` all report `cosine`, `M=16`, `construction_ef=200`, `search_ef=100` (read back with `chroma.collection_settings`) |
| HNSW recall@10 against an exact cosine scan | **1.0** on 40 questions (10 questions asked four times), so `search_ef=100` already returns the exact neighbours at this corpus size |
| HNSW search, per query | p50 **2.25 ms**, p95 **2.94 ms** |
| Exact cosine scan of the same 387 vectors, per query | **15.05 ms** |
| Embedding one question, model loaded and cache empty | **67.6 ms** |
| Embedding the same question again | **0.002 ms** (answered from the cache) |
| Full `vector_store.query` (embed, vector search, word ranking, fusion), cold cache | 38.4–137.1 ms over five distinct questions |
| Full `vector_store.query`, warm cache | p50 **3.21 ms**, p95 **4.79 ms** over 50 queries |

What changed:

- **Query-embedding cache.** The collections are opened with an embedding
  function that remembers the vectors of short texts (512 entries, texts up to
  256 characters). A repeated question skips the model; an upload's long
  passages are not kept.
- **Corpus cache.** The whole-collection read used for the word ranking is
  kept per workspace until the Chroma file changes; the BM25 word counts are
  built once with it. The HNSW tuning itself needed no rebuild: the index was
  already built with `construction_ef=200` and `search_ef=100`, twice and ten
  times Chroma's defaults.
- **Model cache by memory, not by count.** The word engine and teacher caches
  now evict the least recently used model once the weight files pass
  `weights_cache_bytes` (1.5 GiB) instead of dropping every model but one;
  the serving model is never evicted.
- **Incremental startup sync** was already in place and is covered by
  `test_resync_writes_the_workspace_and_only_changed_rows`: a restart writes
  only the FAQ, pending and intent rows whose timestamps moved.

### Which execution provider embeds the text

Chroma's default embedding function hands the ONNX MiniLM the whole provider
list it finds (`CoreMLExecutionProvider`, `AzureExecutionProvider`,
`CPUExecutionProvider`) and pads **every** text to 256 tokens, so the cost of a
batch follows the number of texts, not their length. Measured on this machine
(Apple M4, 10 cores, onnxruntime 1.30.0, arm64 build) by embedding the same
texts twice, once through the default list and once with the CPU only:

| What | Measured |
| --- | --- |
| First batch of one text (creates the session) | **5.23 s** with the default providers, **1.02 s** on the CPU |
| Eight passages of 200 words | **6.15 s** default, **2.52 s** on the CPU — 0.77 s vs **0.32 s** per passage |
| Eight short questions | **4.71 s** default, **3.03 s** on the CPU |
| Largest difference between the two vectors of the same text | **0.0** (384 numbers × 4 passages compared to six decimals) |

`app/services/embeddings.py` therefore asks for `CPUExecutionProvider` and
nothing else: the same model, the same file, the same numbers, about twice the
speed, and no reason to rebuild the index. The test
`test_the_model_is_asked_for_the_cpu_and_nothing_else` keeps the choice. Because
every text is padded to 256 tokens, an upload's cost is proportional to its
passage count — which is also why the duplicate collapse in the same phase is a
speed change as much as a hygiene one.

### The predictor and a prefix longer than its context

The word model has exactly `context` position embeddings — the served checkpoint
reports **64** — and `predict_with_attention` fed it the whole prefix while the
writing path beside it (`writer.py`) already cut the prefix to its last `context`
positions. So a paste of 64 known words, 65 ids once `<start>` is added, raised
`IndexError: index out of range in self` and `POST /language-model/predict`
answered **500**, while a shorter one answered normally — which is why it looked
intermittent. Fixed on 2026-10-28 with the same window the writer uses; the
request schema still accepts 4,000 characters.

Measured on this machine with the served `words-0.pt` (4,147 words, `context` 64,
CPU, 30 timed calls after 3 warm-up, the "before" column run on the same trained
task):

| Prefix | ids fed | before the fix | after the fix |
| --- | --- | --- | --- |
| 3 words | 4 | answered | p50 **0.658 ms**, p95 0.729 ms |
| 48 words | 53 | answered | p50 **0.955 ms**, p95 1.293 ms |
| 65 words (329 characters) | 66 | **500 — IndexError** | p50 **1.106 ms**, p95 1.451 ms |
| 117 words (593 characters) | 118 | **500 — IndexError** | p50 **1.012 ms**, p95 1.291 ms |
| 520 words (2,639 characters) | 521 | **500 — IndexError** | p50 **1.193 ms**, p95 1.725 ms |

The two rows that used to answer are not slower for the change: a prefix of at
most `context` positions is its own window, so the model call is the identical
one. Only the 500 cases became answers, and they cost about the same as the
48-word paste, because the model now reads at most 64 positions whatever was
pasted. The page shows the last 12 of them (`ATTENTION_TOKENS`), and `test_a_prefix_longer_than_the_context_still_predicts`
keeps the guard on a 12-position test model: without the window that test fails
with the same `IndexError`, which was checked before the fix was written.

### What was retired

| What | Measured |
| --- | --- |
| `lessons` Chroma collection | **651** vectors with no metadata, written by the pre-Phase-2 trainer and read by nothing in the runtime; removed on 2026-10-28 by `scripts/retire_artifacts.py`, which first saved all 651 ids, texts and metadata to `backups/retired-20261028-141921-045824/lessons.json` |
| `.language_model/transformer-0.pt` | **10,794,147 bytes** of the retired scratch word model, moved into the same backup folder; `--rollback` puts both back and `tests/test_retire_artifacts.py` covers plan, backup, removal and rollback |
| The collections left after the run | `docs`, `faq`, `intents`, `web` — `docs` still holds **387** vectors |

### How fast an upload is indexed

Measured on 2026-10-28 through the real path on this machine: `POST`-equivalent
`queue_upload` writes the file and the job row, the bounded pool worker reads,
cleans, cuts, collapses and stores it, and the measurement polls the `IngestJob`
row until it is terminal. The text is synthetic and each paragraph has its own
vocabulary, so nothing collapses that would not collapse in real material. Every
file was deleted and its passages removed afterwards, and the store was verified
back at **387** `documents` rows and **387** `docs` vectors.

| Upload | Passages stored | Size | Wall time | Passages per second | Kilobytes per second | Near-duplicates collapsed |
| --- | --- | --- | --- | --- | --- | --- |
| first file after the process starts | 22 | 13,916 B | **8.26 s** | 2.66 | 1.6 | 0 |
| second | 66 | 43,026 B | **13.80 s** | 4.78 | 3.0 | 1 |
| third | 204 | 135,520 B | **33.66 s** | 6.06 | 3.9 | 1 |
| fourth | 473 | 325,177 B | **75.89 s** | **6.23** | **4.2** | 10 |

What the table says: about **6.2 passages per second** and **4.2 KB per second**
once the process is warm, and a little over **4 minutes for a 1 MB text**. The
first file of a process is the slow one per passage — the ONNX session is created
inside that job (1.02 s, see the provider row above) — which is why 22 passages
take 8.26 s while 473 take 75.89 s. Because every text is padded to 256 tokens,
the cost follows the passage count, and the 10 collapses in the last row are 10
passages the run never embedded. No upload left a partial row: the job's own
cleanup and `remaining_rows: 0` in all four measurements.

### How long an answer takes

Measured on 2026-10-28 over HTTP against the running server (`POST /query` with
a bearer token, the workspace's real store: 387 passages, 24 approved answers).
The 24 questions are the 12 first approved-answer questions of the workspace and
the 12 best-scoring stored passages, cut to their first 120 characters. p50 and
p95 are the ordered values at that share of the sample; the run was repeated
after the quality backfill, and these are the second, quieter figures.

| What | Measured |
| --- | --- |
| First request of the process | **1,364.6 ms** — the cost of opening the embedding session and the word engine, which the provider row above measures at 1.02 s |
| Remaining 23 questions, caches cold | p10 46.4 ms, p50 **87.5 ms**, p75 112.2 ms, p95 **122.0 ms**, mean 83.2 ms, range 42.1–195.3 ms |
| The same 24 asked four times, 96 requests, caches warm | p10 21.3 ms, p50 **65.5 ms**, p75 75.4 ms, p95 **92.4 ms**, mean 51.8 ms, range 20.7–95.0 ms |

The warm mean sits below the warm p50 because the path is two populations, and
measuring them separately shows which is which — the median of eight repeats of
each question:

| Question kind | Measured p50 per question |
| --- | --- |
| A question an approved answer owns (12 questions) | **20.9–26.4 ms**, all of them |
| A question that has to search the passages (12 questions) | **66.2–95.4 ms**, all of them |

48 of the 96 warm requests — exactly the four repeats of those twelve approved
answers — finish under 50 ms. The retrieval inside those figures is the cached
path measured above (p50 3.21 ms warm); the rest is the FastAPI request, the
database read and the answer composition.

### How long the process takes to start

| What | Measured |
| --- | --- |
| `import app.main` | **5.62 s**, **5.64 s**, **5.65 s**, **5.88 s** (four runs; the packages pulled in at import are the word model, Chroma and ONNX Runtime) |
| `uvicorn app.main:app` to the first `200` on `/health` | **5.57 s**, **5.61 s**, **5.91 s**, **6.02 s** |
| The same with other work running on the machine | 12.03–15.47 s — measured while the ingest and query runs were live, so the honest figure is the quiet-machine one above |

The steps the app runs itself after importing, timed in one process on the live
workspace:

| Step | Measured |
| --- | --- |
| `init_db` — create the missing tables | 0.123 s |
| `ensure_systemadmin` | 0.485 s |
| `resync_approved_faqs` — the incremental sync of the 24 approved answers | 0.375 s |
| `ingest_jobs.recover` — mark the jobs a restart interrupted | 0.008 s (0 jobs found) |
| open the upload directory | 0.001 s |

All five together are **0.99 s** of the roughly 5.6 s; the rest is importing the
models. The sync is incremental already: on a restart with nothing changed it
writes no row, which `test_resync_writes_the_workspace_and_only_changed_rows`
asserts.

### How much space the stores use

Database `assistance` (PostgreSQL 14 in Docker, port 5433), measured with
`pg_database_size` and `pg_total_relation_size`, row counts from `count(*)`,
after the quality backfill and a `VACUUM ANALYZE`:

| Store | Measured |
| --- | --- |
| Whole database | **9,959 kB** (9,647 kB with the two score columns still empty) |
| `documents` | 856 kB · **387** rows |
| `website_pages` | 208 kB · 14 rows |
| `users` | 96 kB · 1 row |
| `tenants` | 80 kB · 1 row |
| `feedback` | 72 kB · 35 rows |
| `pending_questions` | 72 kB · 21 rows |
| `rules` | 64 kB · 1 row |
| `website_crawls` | 64 kB · 3 rows |
| `faqs` | 64 kB · **24** rows |
| `ingest_jobs` (new this phase) | 40 kB · **0** rows |
| `uploaded_files` (new this phase) | 40 kB · **0** rows |
| `intents` / `invites` | 24 kB each · 0 rows |
| `web_sources` / `web_snapshots` | 16 kB each · 0 rows |

The whole workspace's knowledge — 387 passages, 24 approved answers, 21 inbox
questions, 35 feedback rows — fits in under 10 MB. The two tables this phase
added cost their empty 40 kB each: an ingestion job is a row, not a store, and
everything about the file itself stays in the `documents` rows the job produced.
Nothing was deleted and no index was rebuilt to get these numbers; the growth
from 9,647 kB is the 387 scored rows and the statistics `VACUUM ANALYZE` wrote.

On disk in `assistance_back`:

| Directory | Bytes | What is in it |
| --- | --- | --- |
| `.chroma` | **244,392,442** | 4 collections: `docs` 387, `faq` 24, `intents` 16, `web` 0 vectors |
| `.models` | 999,601,601 | the pinned Luke 0.0.1 checkpoint and its tokenizer |
| `.language_model` | 275,793,568 | the word model, including the retired `transformer-0.pt` until 2026-10-28 removed 10,794,147 B of it |
| `uploads` | 14,895,633 | the 4 PDFs the trainer keeps |

`.chroma` holds far more than 387 vectors because the HNSW index and the SQLite
log grow with every write and are not compacted by a delete; it was
244,206,773 B after the Phase 3 rebuild and vacuum, and has moved 185 kB since.

### What the trainer sees while a file is being read

Every job state was driven in a real browser (headless Chrome over CDP, so the
viewports are the real thing) against the running backend, with the trainer's
own Documents tab. Layout: `.documents-grid` computed **4 columns at 1440 px, 3
at 1024 px (narrowest track 245.328 px) and 1 at 390 px (366 px)**, with no
zero-width element and no horizontal overflow at any of the three. The console
stayed empty apart from the two refusals below, which the server answers with a
415 and a 413.

| State | What the page actually showed |
| --- | --- |
| Idle | `Upload a document` · `csv, xlsx, docx, doc, pdf, txt` |
| Reading | `Indexing…` on the upload card and the same step line on the job card: `Reading the file` → `Finding the passages` → `Storing the passages · 96/406 passages` |
| Progress | sampled 4 s apart: **96/406 at 24 %, 224/406 at 55 %, 352/406 at 87 %** — 128 passages between every pair, so the bar itself moves at about **32 passages a second**; the whole job took ~24 s, of which reading and cutting the file was the first ~8-12 s. That is why the 6.23 passages/s in the upload table above is the slower number: it is measured from the request to the last batch of a real PDF. |
| Done | `406 passages stored` and the toast `Indexed browser-check-mid.txt: 406 passages.` |
| Stopped | red-edged `stopped` card, `Stopped before it finished.`, Retry and Dismiss offered, and the file absent from the document list — the read left no rows |
| Failed | `failed` card carrying the backend's own message, `LibreOffice is required to read this file type` (this machine has no LibreOffice), with Retry and Dismiss |
| Duplicate | `browser-check-long.txt is already in the corpus.` and no second document: 7 documents before, 7 after |
| Refused type | `Unsupported file type: .exe` (415) — no job was started |
| Over the cap | `browser-check-big.txt is larger than 25 MB.` (413) for a 27,262,976 B file |
| Reloaded mid-read | the job card came back still running, at the step the worker had reached |

**What the run found.** After stopping a read and pressing Retry, the job
reported `935 passages stored` but the document card said `1870 chunks`. The
database agreed with the card: 1,870 rows for that source, 935 distinct content
hashes, one document record, and **every `chunk_index` present twice**. The
cause was in this file's own queue: a retry cleared the stop flag of a run that
was still in the file, so the stopped attempt went on to commit its passages
beside the retried run's. Three changes close it — runs of one job now take a
lock for the whole read, a run clears its own stop flag when it takes the job
over rather than the retry doing it, and a terminal report from a run whose job
has since been re-queued is dropped. Fixed with **3 tests**
(`test_a_retried_run_waits_for_the_stopped_one_to_leave_the_file`,
`test_a_retry_after_a_stop_indexes_the_file_once`,
`test_a_late_report_from_a_stopped_run_leaves_a_retry_alone`) and re-measured in
the browser: stop → retry → `406 passages stored`, and the document card and
`/documents` both listed **406**.

**What the verification left behind.** Every test file was removed through the
page's own delete button, and the stores came back to what they were: **387
passages, 387 `docs` vectors**, `faq` 24, `intents` 16, `web` 0, `uploads`
14,895,633 B (the trainer's 4 PDFs), and 0 rows in `ingest_jobs` and
`uploaded_files`. The database itself is 10,101,263 B (9,865 kB) after
`VACUUM ANALYZE`, inside the 9,959 kB recorded above. `.chroma` is
**246,420,848 B — 2,028,406 B larger** than the 244,392,442 B above, and that
growth stayed after the delete: the ~2,700 passages this run added and then
removed cost the HNSW index and the write log about 2 MB that a delete does not
give back.

### What a boundary catch says when it degrades (2026-10-28)

`BLE001` (do not catch blind `Exception`) was added to the Ruff `select` list,
and the first run flagged **23 catches in 11 service modules** that had been
failing quietly: the storage report (8), the optional spaCy grammar model (3),
the taught lesson cards (4), the embedding provider (2), the weather provider
(2), the head and checkpoint files (2), one teacher training run, the generation
worker and the vector index. Each now logs the failure it absorbs and carries a
`# noqa: BLE001 -- <boundary>` directive, and each still returns the same value
it returned before — the same empty list, `None`, `0` or unfinished sentence.

No number in this file was re-measured for it, and the reason is measured rather
than assumed: every new statement sits inside the `except` body, so it runs only
when the fallback was already taken. The four answer-path test files
(`test_answer_flow`, `test_request_answers`, `test_answer_streams`,
`test_answer_finishing`) ran with logging at `DEBUG` — **28 tests passed in
0.95 s and emitted none of the 23 lines**, so the answer path never reaches them.
The whole suite ran the same way it did before the change: **653 tests in
81.61 s**, against 82.51 s beforehand.

What did change is what a trainer can find out. A storage figure that shows 0
because the vector index could not be read now says so at `warning` in the API
log, with the error behind it; a sentence that came back without its article
because the grammar model failed says so at `debug`. Before this, both were
indistinguishable from a workspace that genuinely holds nothing.

## The assessment engine (measured 2026-10-29)

Phase 6 added a measuring stick rather than a new model, and every number below
is read from the rows the engine itself wrote (`scripts/measure_assessment.py`
reports the bank, the scorers, the pilot, the memorisation probe and a finished
run). Nothing was retrained for it; the checkpoint is the same Luke 0.0.1 that
the chat uses.

### The bank it measures against

| What | Measured |
| --- | --- |
| Files | 8 skill files plus `manifest.json`, **431,978 B** in total — **419,486 B** of items and **12,492 B** of manifest, read offline; a run never opens a socket |
| Items | **360** in total: 240 held-out `eval` and 120 `train`, exactly 30 + 15 per skill, `manifest version 1`, split salt `luke-assessment-v1`, **10** recorded source fetches |
| Validation | `bank.check()` returns **no problems**: every item passes `intake.validate_item`, every id is unique, and each skill holds its 30/15 |
| Held-out answer kinds | option **105**, rubric **34**, literal **30**, number **30**, rules **30**, span **11** |
| Instruction families | **19** IFEval rule families in the eval split, from `punctuation:no_comma` (7 items) and `length_constraints:number_sentences` (5) down to `detectable_format:multiple_sections` (1) |
| Sources and licences | conversation ← google/IFEval (Apache-2.0) · creative_writing and reading ← databricks/databricks-dolly-15k (CC BY-SA 3.0), google/boolq, rajpurkar/squad · general_knowledge and science ← allenai/ai2_arc (CC BY-SA 4.0) · grammar ← nyu-mll/blimp (CC BY 4.0) · mathematics ← openai/gsm8k (MIT) · programming ← cruxeval-org/cruxeval (MIT) |
| Per-file digests (SHA-256, first 16 hex) | conversation `9513d83002961cc9` · creative_writing `d9ef26bfc7c84bd7` · general_knowledge `1b6d5fae0654a230` · grammar `4369722be734294c` · mathematics `eb5def67e3d50b37` · programming `a4bee97c574e2a17` · reading `3927d96d633fa4b5` · science `f98db4fb924ce447` |

### What the scorers cost, and the two defects they found

Timed on this machine with 200 repeats over six real items per kind, each answer
marked through the same `scoring.mark` the runner calls:

| Kind | µs per answer | Answered with its expectation | Answered wrong |
| --- | --- | --- | --- |
| option | 1.07 | 1.0 ×6 | 0.0 ×6 |
| number | 1.05 | 1.0 ×6 | 0.0 ×6 |
| span | 3.27 | 1.0 ×6 | 0.0 ×6 |
| literal | 8.32 | 1.0 ×6 | 0.0 ×6 |
| rules | 1.97 | 0.0–1.0, per the rules the answer actually obeyed | 0.0–1.0, never a free pass |
| rubric | 0.32 | `None` ×6 | `None` ×6 |

Scoring is 1–8 µs per answer against a generation that takes seconds, so the
mark is not what a run waits on, and a rubric is never filled in by the code.

Two real defects came out of the pilot, both of which would have understated a
skill in the run below, and both fixed before any score was documented:

- A literal whose expectation the bank stores as its own text (`"[15, 15]"`,
  `"'No such key!'"`) and an answer that wraps its value in backticks or a fence
  scored **0.0** even when it was right. `scoring.py` now reads the expectation
  out of its own text (`_target_literal`), looks for inline code as well as a
  fence (`INLINE`), and falls back to a whitespace- and case-folded comparison
  when only one side parses as a literal (`_spoken`, `_described`). Kept by
  `test_a_literal_the_bank_writes_as_text_is_still_matched` and
  `test_a_literal_string_keeps_its_quotes_on_both_sides`.
- `language:response_language` is deliberately **not ported**: deciding an
  answer's language honestly needs a detector this machine does not have.
  `instruction_rules.check_rules` returns `None` for any rule it cannot check,
  so the **2** eval items carrying that rule read `unassessed` — never failed,
  and never counted as a pass.

### What one item costs: the pilot

12 held-out items taken in turn across the eight skills, generated once each by
the runner's own call path — `conditions.messages`, greedy decoding
(`temperature 0.0`, `max_new_tokens 220`), the installed Luke 0.0.1 — measured on
a quiet machine on 2026-10-29, then marked by the same `scoring.mark` a run uses:

| What | Measured |
| --- | --- |
| Model load in a fresh process | **10.97 s** before the first token of the first item |
| Per-item generation | p50 **1,543 ms**, p95 **59,245 ms**, mean 13,147 ms, min **177 ms**, max **70,178 ms** |
| Length of what came back | mean **67.4** new tokens; **1** of 12 ran out of the 220-token budget (`stop_reason = max_new_tokens`), the other 11 ended at `eos` |
| Decoding speed | **5.13** new tokens per second, counted over the whole pilot rather than per item |
| Retrieval read (assisted mode only) | p50 **2.6 ms**, p95 **10.3 ms** — but the first read of the process is **1,633 ms** because it opens the embedding session, and **0 hits** on all 12 items (see the store row below: the passage collection is empty in this workspace today) |
| Extrapolation to the whole held-out split | the pilot's mean predicted **3,155 s** for 240 items; the real run took **2,205 s**, so a 12-item pilot overstates a full run by about 43 % and is a sizing guide, not a promise |

The spread is the point: a 2-token grammar answer costs 177 ms and a story that
fills the budget costs 59 s, so a run's wall time is set by what the model
decides to say, not by the number of items.

### The unaided run of the whole held-out bank

Run 2, the complete 240-item eval split, `mode = unaided`, greedy decoding
(`temperature 0.0`, `max_new_tokens 220`), on the installed checkpoint:
`Luke 0.0.1`, artifact `Qwen/Qwen2.5-0.5B-Instruct@7ae5576…#fdf756fa7fcbe740`,
tokenizer `Qwen2.5-0.5B-Instruct@7ae5576:tokenizer.json:c0382117ea329cdf`,
bank v1.

| What | Measured |
| --- | --- |
| Items answered | **240 of 240**, status `done`, **no empty response** stored |
| Wall time | **2,205 s** (36.8 min) |
| Per-item generation latency | p50 **1,659 ms**, p95 **13,808 ms**, mean 9,098 ms, min 322 ms, max **282,281 ms** — the tail is an item that filled the 220-token budget |
| Scored / unassessed | **204 scored**, **36 unassessed**: 34 rubric answers waiting for a trainer, 2 items whose only uncheckable rule is `language:response_language` |
| Mean score of what is scored | **0.4158**; **82** of 204 outcomes reached the 0.8 pass mark |

| Skill | Scored | Mean | Passed | Waiting | Confidence | Stage |
| --- | --- | --- | --- | --- | --- | --- |
| grammar | 30 | 0.5667 | 17 | 0 | high | none cleared |
| science | 30 | 0.5333 | 16 | 0 | high | none cleared |
| general_knowledge | 30 | 0.5000 | 15 | 0 | high | none cleared |
| reading | 26 | 0.4231 | 11 | 4 | high | none cleared |
| conversation | 28 | 0.3869 | 8 | 0 | high | none cleared |
| mathematics | 30 | 0.3000 | 9 | 0 | high | none cleared |
| programming | 30 | 0.2000 | 6 | 0 | high | none cleared |
| creative_writing | 0 | not measured yet | 0 | 30 | none | nothing scored |

What the shape of that table says, in the engine's own terms: seven skills have a
real number measured from 30 held-out items each, so all seven read `high`
confidence; none of them cleared stage 1, which needs a stage mean at or above
0.8 over at least three items, so every card says `stage: none` and the honest
`next_step` (practise the topic that failed). Creative writing is `unassessed`
with a null score rather than 0.0 because its 30 rubric answers need a trainer
and the checkpoint does not judge its own stories. `delta` is empty for every
skill: the only earlier unaided run is run 1, which the machine interrupted at
30 of 240 items, and a partial run is not a previous measurement.

Run 1 is also the reason the tables keep their history honestly. Docker Desktop
exited mid-run, Postgres refused connections, and the worker died with it;
`runner.recover()` marked that run `failed`, kept its **30** stored grammar
results (0.5667, the same figure run 2 measured for those items — greedy
decoding repeated them exactly), and left them out of every profile comparison.

### How much of this is memory

The base checkpoint's pretraining corpus cannot be audited on this machine, so
contamination is measured on the items instead: the same item, with the task
framing and the options removed, and the answer looked for as its own word in
what comes back.

| What | Measured |
| --- | --- |
| Probes | **12** held-out items, spread across the skills, each with an answer long enough to survive inside prose (`MIN_RECALL_ANSWER_CHARS = 5`) and no longer shown in the prompt; **17** further items were left out for failing that test |
| Answers named from memory | **2 of 12 = 0.1667** — and both were one-word options (`direction`, `mountain ranges`) that a plausible sentence can contain by accident |
| What the other 10 did | answered around the question without naming its option, restated the reasoning, or (the two programming items) explained the code instead of printing its literal |
| Mechanical ban | no `eval` item can reach training material at all: `integrity.is_held_out` drops it in `language/writing.py`, `language/corpus.py` and `language/training_materials.py`, and `tests/test_assessment_integrity.py` plus `tests/test_assessment_loader.py` assert that no held-out id or prompt reaches a loader or the tokenizer corpus |

So the low skills are not a memorisation failure and the scores are not a
memorisation success: with 17 % of probes naming their answer, the honest reading
is that these numbers measure what the checkpoint does with a task, not what it
remembers of a benchmark.

### The same questions through the chat path, in both modes

Measured 2026-10-29 on a quiet machine over HTTP against `uvicorn app.main:app`
with a bearer token, workspace 0: the 12 first approved-answer questions asked
four times each (48 requests) and the 6 inbox questions asked twice each (12
requests), the whole set run once with `"retrieval": true` and once with
`"retrieval": false`. The first request of each set is reported apart because it
opens the embedding session.

| What | Retrieval on | Retrieval off (`unaided`) |
| --- | --- | --- |
| A question an approved answer owns (48) | first **1,858.8 ms**; then p50 **39.9 ms**, p95 **67.4 ms**, mean 42.7 ms | first **3,578.1 ms**; then p50 **1,625.4 ms**, p95 **7,200.5 ms**, mean 2,676.1 ms |
| A question from the inbox (12) | p50 **204.4 ms**, p95 **39,836.8 ms** | p50 **4,651.6 ms**, p95 **11,228.1 ms** |
| What answered it | `retrieved/faq`, `retrieved/lesson_cards`, `generated/generated` | `generated/generated` only |

This is the mode switch doing its job and costing what it costs: the same twelve
questions that answer in 40 ms from stored material take 1.6 s when the model has
to say them itself — a 41× median difference, and every unaided answer is labelled
`generated`. The 39.8 s p95 on the retrieval side is a single inbox ask that
matched nothing and ran out to the full generation budget; its median is 204 ms.
None of the 24 inbox asks wrote to the trainer's inbox: `pending_questions` held
**21** rows before the run and **21** after, with no row's `last_seen_at` later
than 2026-10-27.

### How long it starts and what it takes up

| What | Measured |
| --- | --- |
| `import app.main` | **4.96 s**, **5.26 s**, **5.88 s**, **6.19 s** (four runs) |
| `uvicorn app.main:app` to the first `200` on `/health` | **6.61 s**, **4.73 s**, **7.56 s** (three starts) |
| Whole database `assistance` | **10,026,007 B** (9,791 kB; Phase 5 recorded 9,959 kB) |
| `assessment_results` (new this phase) | **245,760 B · 270 rows** — 240 from run 2 and 30 from the interrupted run 1 |
| `assessment_runs` (new this phase) | **32,768 B · 2 rows** |
| `faqs` / `pending_questions` / `feedback` | 65,536 · 24 rows / 73,728 · 21 / 73,728 · 35 |
| `website_pages` / `documents` | 212,992 · 14 rows / 212,992 · **0** rows |
| `.chroma` | **248,100,948 B** — 4 collections: `docs` **0**, `faq` 24, `intents` 16, `web` 24 vectors |
| `.models` | 999,601,601 B — the pinned Luke 0.0.1 checkpoint and its tokenizer, unchanged |
| `.language_model` | **277,092,805 B** — the word model and its state, including **1,033** learned lesson cards |
| `uploads` | **0 B** |
| `app/data/assessment` | **431,978 B** — the nine bank files, the only files this phase added |

A full run is a row and 120 result rows per skill-run; 270 answer rows — 240 of
them with their text — cost 246 kB, which is why a run can be kept forever
instead of pruned. `documents` still occupies 212,992 B while holding no rows
because nothing was vacuumed after its contents left; the database total is
smaller than Phase 5's in spite of the two new tables for the same reason.

### What the workspace holds today, and what that does to these numbers

Between the Phase 5 row above and this one the workspace's passage store emptied:
`documents` went from 387 rows to **0**, the `docs` collection from 387 vectors to
**0**, and `uploads` from 4 PDFs (14,895,633 B) to **0 B**. Nothing in the
assessment engine deletes stored material — the only ways passages leave are the
per-file `DELETE /documents/{filename}` route and the index-rebuild script, and
neither was run for Phase 6.

What follows for anyone reading these two phases together:

- The Phase 5 retrieval and answer-speed rows stay true of the workspace they
  were measured in; they are not reproducible here today.
- Today, assisted mode reads its evidence from the **24** approved answers and
  the **1,033** lesson cards and reads **0** passages, so the pilot's retrieval
  row above shows 0 hits on every item.
- The assessment figures are unaffected either way: the run that produced them was
  `unaided`, which by design reads no stored material at all.

## The adapter compatibility run (measured 2026-10-29)

Phase 7 has not been built. This is the install check the plan asks for before it
is: can `peft` train and serve a LoRA adapter on the checkpoint Luke 0.0.1 is
installed, inside the pinned environment, without disturbing it? One scratch
script, run against the real artifact folder
`.models/foundation/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775`
with `local_files_only`, nothing written to the workspace, and the saved adapter
going to a temporary directory outside it which was removed once the numbers were
read. Not the app's request path, so none of the chat or assessment rows above are
reproduced here.

| What | Measured |
| --- | --- |
| Environment after the install | `torch 2.14.1`, `transformers 5.20.2`, `peft 0.21.2`, `accelerate 1.15.0`, `psutil 7.2.2`; the freeze diff shows exactly those three additions and no pin moved |
| Backend gate with `peft` present | `make quality` green — **783** tests pass, structure check and format check unchanged |
| Serving load, the path the app uses | MPS, float16, **0.45–0.94 s** to load the base |
| LoRA, CPU float32 | `r = 4` over `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`: **2,199,552** trainable of **496,232,320** = **0.4433 %** |
| Two real optimiser steps | losses **2.8239** then **1.7427**; **0.32 s** and **0.26 s** per step (forward, backward, AdamW update) |
| Saved adapter | **8,848,215 B** in three files: `adapter_model.safetensors`, `adapter_config.json`, `README.md` |
| Reload and serve the adapter | `PeftModel.from_pretrained` on top of the float16/MPS base: **0.36 s** to attach, then a greedy 24-token answer |
| LoRA backward on MPS in float16 | **works** — one step, loss **2.8225**, no CPU fallback needed for this model and this `r` |
| Memory of the whole script | peak resident set **4,679,991,296 B** (≈ 4.4 GiB), which covers four loaded copies of the base |

Two findings worth more than the numbers:

- **The module names changed.** The first attempt targeted `c_attn` and `c_proj`,
  the fused QKV and projection names Qwen2 used to have, and `peft` refused:
  `NoMatchingPeftModuleError: Target modules {'c_attn', 'c_proj'} not found in
  the base model`. Under `transformers 5.20.2` the checkpoint's layers expose
  `q_proj, k_proj, v_proj, o_proj` inside `self_attn` plus
  `gate_proj, up_proj, down_proj` inside `mlp`. Any Phase 7 training code must
  take the names from the model it loaded, not from a recipe.
- **Generation speed through this script is not a number to quote.** The same
  base-model serve in the same script measured **13.41** tokens/s on one run and
  **42.86** tokens/s on another, and the adapter-attached serve measured **10.04**
  tokens/s in the run where the base measured **42.86**. That spread is the MPS
  kernels warming up inside a fresh process, not a property of the adapter, so the
  honest statement is that adapter overhead versus base is **not measured yet** —
  it needs re-measuring through the app's own runtime, which keeps one model warm.

What Phase 7 still has not answered, and cannot from this run: whether an adapter
beats the baseline on the Phase 6 held-out bank. That comparison is the shipping
gate for any adapter. Both halves are now measured — the baseline is the next
section, the candidate is the gate section below — and the answer is no.

## The baseline an adapter has to beat (measured 2026-10-30)

Run 3, `scripts/measure_assessment.py --mode assisted --tenant 0`: the complete
240-item held-out split through the path the app ships, retrieval allowed, no
adapter, on the installed checkpoint. Nothing about the model, bank or decoding
config was changed since run 2 — this is a re-measure, not a new setup.

| What | Measured 2026-10-30 |
| --- | --- |
| Run | **3**, `mode = assisted`, **240 of 240** items answered, status `done`, no timeouts |
| Checkpoint | `Luke 0.0.1` · `Qwen/Qwen2.5-0.5B-Instruct@7ae5576…#fdf756fa7fcbe740` · bank v1 · `temperature 0.0`, `max_new_tokens 220` |
| Mean score of what is scored | **0.4158** over **204** items; **82** outcomes reached the 0.8 pass mark; **36** unassessed (34 rubric answers waiting for a trainer, 2 items whose only rule is `language:response_language`) |
| Per-skill means | **all eight rows identical to run 2** — grammar 0.5667, science 0.5333, general_knowledge 0.5000, reading 0.4231, conversation 0.3869, mathematics 0.3000, programming 0.2000, creative_writing unassessed |
| Responses | **240 of 240 byte-identical to the unaided run 2**, compared in the database item by item |
| Evidence retrieved | **0 of 240** items were given any evidence at all |
| Wall time | **459.2 s** (run 2: 2,205 s) |
| Per-item generation latency | p50 **514 ms**, p95 **6,642 ms**, mean 1,876.8 ms, min 88 ms, max 7,219 ms (run 2: 1,659 / 13,808 / 9,098 / 322 / 282,281 ms) |

Two findings, and the first one changes what the gate means:

- **On this workspace the assisted run is not a different measurement.** Every
  response is the one the unaided run wrote, because `_evidence()` returned
  nothing for all 240 prompts: the `docs` collection holds **0** vectors, since
  `documents` has **0** rows and `uploads` is **0 B** (the Phase 6 and dataset
  sections above record the same emptiness). So the number an adapter must beat
  today is **0.4158 over the 204 auto-scored items**, and it is a
  retrieval-assisted baseline in name only — it measures the checkpoint alone.
  The honest consequence for Phase 7: a promotion decided against this number
  tests whether the model learned, which is what the gate asks, but it cannot
  say anything about retrieval, and the true retrieval-assisted baseline stays
  **not measured yet** until a real document set is uploaded.
- **The scores repeat exactly; the timings do not repeat at all.** Same 240
  items, same greedy config, same artifact: 240 identical strings and the same
  mean, which is the determinism the runner claims for itself. The same run took
  a fifth of the wall time of run 2 because this script loads the model once and
  keeps it warm for all 240 items, while run 2 paid a **10.97 s** cold load and
  a colder MPS. A score travels between runs; a timing belongs to the process
  that measured it, and is recorded here as a property of run 3 only.

## The Phase 7 dataset, measured on the workspace (2026-10-30)

`scripts/measure_dataset.py --tenant 0` reads the corpus through
`app/services/training/dataset.py` — the same reading the word model trains
from (`training_materials.gather`), tagged with the split — and reports what the
Phase 7 operations would actually be given.

| What | Measured 2026-10-30 |
| --- | --- |
| Material read | **1,364** pieces, of which **14** give no example (1 words card, 13 written sentences) |
| Sources | **1,062** — 837 tagged train, 225 held out |
| Split units | **1,008** — **46** groups cover **100** sources that share an answer word for word |
| Examples | **1,350** — `sft_train` 1,124, `sft_held_out` 226 |
| Passage examples (continued pretraining) | **0** — `documents` holds **0** rows and `uploads` is **0 B** |
| Leakage after grouping | **0** texts on both sides (it was **13** before) |
| Split | salt `luke-training-v1`, share `0.2`, taken: yes |
| Time to build | **2.4 s** for the whole workspace |

Two findings:

- **The corpus repeats itself, and a source split alone does not catch it.** The
  first build reported 13 held-out sources whose exact answer text was also on
  the training side — pairs of starter cards with the same answer ("Describe a
  circle." and a neighbour card), and approved answers that restate a taught one
  (`Who are you?`, `What are the rudiments?`). Training one copy and measuring
  the other reads as ability and is memory, so the split unit is no longer the
  source but the transitive group of sources sharing a text
  (`dataset._units_of`): the 1,062 sources fall into 1,008 units, the largest
  group holding five of them (the Twinkle poem as an approved answer, two
  corrections and a taught lesson), and the leak count went **13 → 0**.
  `leakage()` is kept as the check a run makes before it starts.
- **There is nothing to continue-pretrain on yet.** Every example the workspace
  can give is a question with an answer, because the passage store is empty
  (Phase 6 recorded the same `documents` 387 → 0). The SFT side is measurable
  today; the continued-pretraining side needs a real document set uploaded
  first. The shipping gate has the same shape: the baseline half is the
  **0.4158** recorded above, and the adapter half is now measured — **0.1021**,
  which does not beat it, so the adapter is not promoted. Until documents exist
  that comparison still says nothing about retrieval.

The same command re-run alongside the adapter run below reported **1,353**
examples (1,127 train, 226 held out) from the same **1,062** sources: three
further training examples were recorded in the store after the first build, and
neither the held-out side nor the leak count (**0**) moved.

## The first adapter run (measured 2026-10-30)

`scripts/measure_training.py --operation lora --tenant 0 --steps 20`, run twice
in one session. The script reads the dataset above, refuses an empty or leaking
held-out side, loads the installed base model, wraps the linear layers the
loaded model really has, runs the shared loop, and writes the adapter through a
staging folder. The figures are the second run; the first is compared at the
end.

| What | Measured 2026-10-30 |
| --- | --- |
| Operation | `lora` → kind `adapter`, `written: true`, `stop = steps`, device `mps` |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (installed checkpoint), `peft 0.21.2` |
| Dataset | **1,062** sources, **1,353** examples — **1,127** training rows from **837** sources, **226** held out and untouched, leakage **0** |
| Shape | **20** steps, batch **2**, lr **5e-4**, max **512** tokens, seed **0**, rank **8**, alpha **16**, dropout **0.05** |
| Trainable | **4,399,104** of **498,431,872** parameters — **0.8826 %** |
| Wrapped layers | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` — read from the loaded model, not from the `c_attn`/`c_proj` recipe |
| Loss | **1.4728 → 0.8196 → 0.6690 → 0.7440** at steps 5/10/15/20; mean of the last five **0.7440** |
| Time | **11.5 s** inside the loop, **19.2 s** wall including load and write; peak RSS **1,041,891,328 B** |
| Artifact | **17,646,523 B** — `adapter_model.safetensors` 17,640,136 B, plus `adapter_config.json` and README — written atomically; the tokenizer was not touched |

Three findings:

- **A run repeats only when the seed is set before the adapter exists, and
  again for the loop.** The first two script runs, with no seed anywhere, ended
  at **0.7405** and **0.7230**. Seeding only inside the loop — after the adapter
  had already been wrapped — made the disagreement worse: **1.8595 vs 0.7052**,
  because each process drew different LoRA matrices and trained different
  starting adapters. Seeding at the start of the operation and again at the
  start of the loop gave two runs whose four report lines agree to four
  decimals and whose artifacts are the same **17,646,523 B**; only wall time and
  peak RSS moved (19.1 vs 19.2 s; 1,050,542,080 vs 1,041,891,328 B).
- **A falling loss is not a promotion.** The loss falls on the rows it trained
  on; the gate is the **0.4158** baseline on the held-out Phase 6 bank, and this
  adapter has not been put through it. That half of the comparison is **not
  measured yet**, and 20 steps over 1,127 rows is less than one pass over the
  corpus, not a converged teacher.
- **The other two operations were not run against the real store.** Continued
  pretraining is refused by name because the passage store holds **0** rows, and
  a whole-weight SFT run would overwrite the served checkpoint artifact, so its
  measurement waits for the promotion decision rather than being run for a
  number. Both are exercised by tests on a small model.

## The adapter gate (measured 2026-10-30)

The candidate is the adapter from the section above —
`measure-lora-1793338933#771ea84edc1f4dea`, rank 8, 20 steps, the same weights
whose training loss fell to **0.7440**. It was attached to the installed
checkpoint through `scripts/measure_assessment.py --mode assisted --tenant 0
--adapter …`, which loads it through the app's own `runtime` and records its
weight hash in the run provenance. Run 5 is the measurement.

| What | Baseline (run 3) | Candidate (run 5) |
| --- | --- | --- |
| Checkpoint | base `fdf756fa7fcbe740` | base + `adapter:measure-lora-1793338933#771ea84edc1f4dea` |
| Items answered | 240 of 240, `done` | 240 of 240, `done`, no timeouts |
| Mean score of what is scored | **0.4158** over **204** | **0.1021** over **204** |
| At the 0.8 pass mark | **82** | **17** |
| Unassessed (rubric) | 36 | 36 |
| Wall time | 459.2 s | 298.2 s |
| Per-item latency | p50 514 / p95 6,642 ms | p50 695 / p95 3,845 ms |
| Responses unlike the baseline | — | **238 of 240** |

Per skill, scored items only:

| Skill | Baseline | Candidate |
| --- | --- | --- |
| conversation | 0.3869 | **0.3155** |
| general_knowledge | 0.5000 | **0.0333** |
| grammar | 0.5667 | **0.2333** |
| mathematics | 0.3000 | **0.1000** |
| programming | 0.2000 | **0.0000** |
| reading | 0.4231 | **0.0000** |
| science | 0.5333 | **0.0333** |
| creative_writing | unassessed | unassessed |

**Decision: the adapter is not promoted.** It loses **0.3137** of mean score
against the baseline and falls in every skill it was scored on. Nothing in the
app serves it by default: `--adapter` is a measurement path, and promotion would
be a separate, recorded change.

Three findings:

- **The failure is answer shape, not silence.** The adapter learned the lesson
  style it was trained on. Where the base answered option items with `B`, the
  adapter wrote "The correct sentence is B." — correct in substance, unscorable
  as an option — and reading fell from **0.4231 to 0**. That is visible in the
  238 responses that differ from the baseline; none were empty.
- **The first gate attempt was invalid and was corrected.** Run 4 recorded the
  adapter in its provenance but answered with the base: the model cache treated
  the runner's plain `load()` as a request to reload without the adapter. Every
  response was byte-identical to run 3, which is what exposed it. The cache now
  keeps an explicitly attached adapter until a different one is named
  (pinned by `tests/test_runtime_adapter.py`), run 4's artifact was corrected to
  the base checkpoint it actually used and marked as superseded, and run 5 is
  the measurement above.
- **A falling training loss still is not a promotion.** The candidate's loss
  fell on its own rows over 20 steps — less than one pass — while the bank it
  was never trained on fell with it. The gate is the only number that decides.

## Adapter serving cost and the scratch word model (2026-10-30)

- **No adapter is served, so its warm overhead has no number.** The only
  candidate failed the gate above, and nothing in the app loads it outside the
  `--adapter` measurement path. If one is ever promoted, its overhead has to be
  measured warm on identical prompts before it serves; that measurement does
  not exist yet and is not claimed.
- **The gate run's latency is not an overhead figure.** The candidate wrote
  different text — 238 of 240 responses differed, and the adapter answered
  option items with whole sentences where the base answered with a letter — so
  p50 695 ms against p50 514 ms compares different generated lengths and work,
  not the same work with and without the adapter.
- **The scratch word model stays an educational opt-in.** Training uses the
  word tokenizer by default; byte-level BPE has to be chosen explicitly, and a
  BPE checkpoint is labelled scratch-trained. The scratch model predicts the
  next word and the attention it measured — it never writes a chat answer,
  which comes from the installed checkpoint, a chosen teacher, or a taught
  answer.

## Keeping this file honest

- **Updated on every improvement**: any change to the brain — the word model,
  the training loop, the heads, the request path — that could move these
  numbers re-measures per the drill and adds a row (or updates one) in the same
  change, before the work is called complete. The rule is written into
  `../../CODE_STANDARDS.md`.
- **Updated on every retrain**: when the trainer retrains the brain for
  themselves, the new row belongs here too — the panel numbers and this file
  must agree.
- **No fakes**: a missing measurement is written as "not measured yet", never
  as a number. A panel number that cannot be measured does not ship.
