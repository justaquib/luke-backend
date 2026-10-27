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
