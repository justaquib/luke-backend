# The Model's Measurements — Real Numbers Only

> One place for what Luke's tiny model actually measures, kept current after
> every improvement. Every number here was measured on this machine against
> the material the app really trains on. Nothing estimated, extrapolated or
> "would probably be" — the strict rule in `../../CODE_STANDARDS.md` forbids
> that. How the frontier labs measure the same qualities:
> `docs/06-frontier-model-evaluation.md`.

## The measuring drill

Every row below was taken the same way, so the rows can be compared:

1. Build the training set exactly as production does (`training_materials.collect`,
   `quality.keep`, `example_service.build`), with the same vocabulary the model
   actually holds.
2. Read the model in its resting state (`eval` mode, dropout off, `torch.no_grad`)
   over fixed, seeded batches — the same read `training._check` does.
3. Report: the averaged loss over those batches, the number of examples behind
   it, and how many passes the model had over that material.

Chance level for orientation: `ln(vocabulary)` is what pure guessing scores.
A loss of L means the model, at each word, puts about `e^-L` of its chance on
the exactly right token, or put the other way, hesitates between roughly `e^L`
equally likely words.

## The table

| When | Model state | Loss (true, averaged) | Panel said | Passes over the material | Chances the right word | Vocabulary | Examples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23 | Saved writer, trained by the old loop — reported `loss = one random batch of 16`, steps 2700 ≈ 1.2 passes | **2.84** | 2.4127 (noise) | ~1.2 | ~5.8% | 2092 | 36,374 |
| 2026-10-23 | Measured run, measured before the fix — same as production was: flat 3000 steps, batch 16 | **~6.8 at step 3000** | (not shown; the panel number was a single batch read) | ~0.7 (curriculum keeps the hard half out until late) | ~0.1% | 1644 | 36,374 |
| 2026-10-23 | Measured run after the fix — 9000 steps, batch 16 ≈ 4 passes; still falling when it hit the run's end | **1.377** | — | ~4.0 | ~25% | 1644 | 36,374 |
| Chance level | Guessing | 7.65 (`ln 2092`) / 7.40 (`ln 1644`) | — | — | ~0.0% | — | — |

What the rows say in one line each:

- The saved model the trainer saw on the panel was described as **2.4127** but
  truly knows at **2.84** — the panel number was one unlucky batch, not the
  model.
- The old flat-3000-step training was materially too short: at step 3000 the
  model hesitated among `e^6.8` ≈ 900 words — barely better than guessing.
- Four passes took it to **1.377** and it was *still* falling at the end, so
  the honest lever next is more passes (`TRAINING_PASSES`), not a bigger
  model.

## What the model does with what it knows

Measured behaviour on real requests, with the guards built this session:

| Change | Behaviour before | Behaviour after |
| --- | --- | --- |
| Loop guard (`quality.repeats_itself`, `MAX_PAIR_SAYS = 2`) | "write a poem on rain" → `rain, go away, go away, go away, go away` (grounded, so given away) | the writer's loop is refused; the full taught poem is given instead (verified against the live model) |
| Kind-scoped material search (`requests.find_material`) | "write a poem on star" refused even with the poem taught — "star" is said by 8 fact lessons, so the topic index dropped the word | the kind the request asked for picks the pool; the poem is matched and answered (verified with a Twinkle lesson added) |
| Truthful checkpoints + pass-based training (`training._check`, `TRAINING_PASSES`) | panel showed a single batch's loss; early stopping decided on that noise; the training saw the material ~1.2 times | panel shows the averaged resting-state read; stopping decides on that; the training makes 4 passes |
| Stored instruction measurement (`brain.remeasure_instruction`, `instruction-{tenant}.json`) | the instruction test (12 request answers, 8 refusals, a 400-step probe training) re-ran on every brain read — a brain page load took ~59 s | a brain page load reads the stored measurement in ~1.5 s; a fresh measurement is taken only when the trainer presses the test button, or a training finishes, and it is stored with its time |
| Level ladder in hundredths with an approval gate (`brain.LUKE_LEVELS`, `approve_level`) | the level auto-advanced to whatever the measurements said and borrowed model-level names for the volume scale | the ladder is 0.0.0 → 0.0.9 → 0.1.0; the measured level and the displayed level are separate numbers and the trainer approves a promotion; the maturity scale keeps its own human stage names |

## The frontier comparison — in kind, not in number

You cannot put Luke's 1.38 next to a frontier model's published loss and rank
them: the numbers sit on different corpora, different tokenizers, different
vocabularies. The honest comparison is the *discipline* of measurement, which
follows the same conventions (details in
`docs/06-frontier-model-evaluation.md`):

| The frontier convention | Ours (measured where shown) |
| --- | --- |
| Held-out, averaged, eval-mode loss — never a single batch | `training._check`: 10 fixed batches, resting state — that is the 2.84 and the 1.377 above |
| Perplexity declared only against its own corpus/tokenizer | the `e^L` column: 17 words of doubt → 4 |
| Scaling laws: length of training measured against the material | steps of 0 = 4 passes over the examples, capped at 12000 steps |
| Grounding: every sentence supported by the given material | `_grounded`: every word from the taught material, else the taught answer |
| Calibration: "I am not sure" must mean something | the token-confidence critique and the unsure-word note |
| Refusal instead of invention about the unknown | the taught-material boundary (refused request, never a guess) |
| Contamination control: probes held out of training | the instruction probes in `language/evaluation.py` |

## Keeping this file honest

- **Updated on every improvement**: any change to the brain — the writer, the
  training loop, the heads, the request path — that could move these numbers
  re-measures per the drill and adds a row (or updates one) in the same
  change, before the work is called complete. The rule is written into
  `../../CODE_STANDARDS.md`.
- **Updated on every retrain**: when the trainer retrains the brain for
  themselves, the new row belongs here too — the panel numbers and this file
  must agree.
- **No fakes**: a missing measurement is written as "not measured yet", never
  as a number. A panel number that cannot be measured does not ship.
