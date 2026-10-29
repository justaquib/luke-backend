# Assessment item bank (Luke Lab)

The bank is what the assessment engine measures against. It is versioned,
vendored and offline-readable: eight JSON files under
`app/data/assessment/`, one per skill, plus a manifest that records where every
item came from and what each file contains. No assessment run touches the
network.

## What is in it

Eight skills, **45 items each — 360 in total**:

| file | skill | items | held out | training | kinds |
| --- | --- | --- | --- | --- | --- |
| `grammar.json` | Language comprehension and grammar | 45 | 30 | 15 | `option` |
| `conversation.json` | Conversation and instruction following | 45 | 30 | 15 | `rules` |
| `reading.json` | Reading comprehension | 45 | 30 | 15 | `span`, `option`, `rubric` |
| `creative_writing.json` | Creative writing | 45 | 30 | 15 | `rubric` |
| `mathematics.json` | Mathematics | 45 | 30 | 15 | `number` |
| `programming.json` | Programming | 45 | 30 | 15 | `literal` |
| `general_knowledge.json` | General knowledge | 45 | 30 | 15 | `option` |
| `science.json` | Science | 45 | 30 | 15 | `option` |

The eight skill files are 419,486 bytes; with the manifest the directory is
431,978 bytes. Each skill file is
`{"skill", "title", "items": [...]}`, items sorted by id so a rebuild diffs
cleanly.

Every item carries the same shape:

```json
{
  "id": "mathematics:gsm8k:12",
  "skill": "mathematics",
  "stage": 2,
  "prompt": "...",
  "topic": "rates",
  "expected": {"kind": "number", "answer": 9},
  "split": "eval",
  "source": {
    "name": "gsm8k",
    "dataset": "openai/gsm8k",
    "record": "12",
    "licence": "MIT",
    "url": "https://huggingface.co/datasets/openai/gsm8k"
  }
}
```

`expected` is per kind: `answer` plus `choices` for `option`, `answer` for
`span`/`number`/`literal` (with `acceptable` variants for a span), `rules` for
`rules`, and `criteria` plus an optional human `reference` for `rubric`. The id
is `skill:source:record`, so an item can always be traced back to the row it
was built from. `stage` is 1–3.

## Where it comes from

Ten licensed sources. Licences were read from each dataset's card at build time
and copied into every item and into the manifest — the table below is the
record, not a guess:

| source | skill | dataset | licence | rows read | items shaped |
| --- | --- | --- | --- | --- | --- |
| `blimp` | grammar | `nyu-mll/blimp` | CC BY 4.0 | 360 | 360 |
| `ifeval` | conversation | `google/IFEval` | Apache-2.0 | 200 | 200 |
| `squad` | reading | `rajpurkar/squad` | CC BY-SA 4.0 | 300 | 300 |
| `boolq` | reading | `google/boolq` | CC BY-SA 3.0 | 300 | 300 |
| `dolly_summary` | reading | `databricks/databricks-dolly-15k` | CC BY-SA 3.0 | 1,500 | 106 |
| `gsm8k` | mathematics | `openai/gsm8k` | MIT | 300 | 300 |
| `cruxeval` | programming | `cruxeval-org/cruxeval` | MIT | 400 | 400 |
| `arc_easy` | general_knowledge | `allenai/ai2_arc` | CC BY-SA 4.0 | 300 | 300 |
| `arc_challenge` | science | `allenai/ai2_arc` | CC BY-SA 4.0 | 300 | 300 |
| `dolly_creative` | creative_writing | `databricks/databricks-dolly-15k` | CC BY-SA 3.0 | 1,500 | 76 |

BLiMP contributes 12 paradigms × 30 rows, so grammar items are spread over
agreement, argument structure, islands, ellipsis, raising, scope,
reconstruction and NPI licensing rather than one paradigm. ARC contributes both
difficulty levels: Easy feeds general knowledge, Challenge feeds science. Dolly
is sampled evenly across its 15,015 rows; only its `summarization` and
`creative_writing` categories survive, which is why 1,500 rows shape 106 + 76
items. Items a shaper cannot build honestly (missing answer, identical pair,
unparsable code, mismatched instruction kwargs) are dropped rather than
patched.

The `reading` skill deliberately mixes three kinds — SQuAD spans, BoolQ yes/no,
and Dolly summaries — so a reading score is not one test's score.

## The split rule

Each skill's pool is ordered by `sha256("<salt>:<item_id>")` and cut at fixed
sizes: the first **30** items are held out for assessment, the next **15** are
for training and trainer review. The salt is `luke-assessment-v1`, recorded in
the manifest. Nothing is random at build time, so the same pool yields the same
split regardless of input order, and a different salt yields a different split
— both are pinned by tests.

An item's split is part of the stored item, not computed at load time. Moving
an item between splits therefore requires a deliberate rebuild, and the reason
to rebuild (a new salt) is visible in the manifest.

## Building and checking

```bash
cd assistance_back
.venv/bin/python scripts/build_item_bank.py            # rebuild, needs network
.venv/bin/python scripts/build_item_bank.py --check    # offline verification
```

The builder reads each source, shapes drafts, assigns splits, validates every
stored item, and writes each file atomically (`.tmp` then replace) plus the
manifest with a SHA-256 and byte count per file. `--check` reads only local
files: it verifies the digest, title and counts of each file, re-runs
`validate_item` on every item, and exits non-zero if anything disagrees. It
prints `(checked)` and the same table shown above; it never fetches.

Two operational notes:

- The Hugging Face rows server is fronted by CloudFront and answers bursts with
  `429` and no `Retry-After`. The builder retries `429/500/502/503/504` up to
  six times with exponential backoff, keeps 1.5 s between requests, and reads
  IFEval, CruxEval and Dolly from their raw files at pinned revisions
  (`ifeval_input_data.jsonl` @ `966cd89`, `test.jsonl` @ `b96af04`,
  `databricks-dolly-15k.jsonl` @ `bdd27f4`) instead of through the paginated
  API. A rebuild should be treated as a paced background job, not a tight loop.
- Every fetch — config, split, row count, whether rows are spread, file and
  revision — is recorded in `manifest.json`. A rebuild that changes any of them
  produces a new manifest, which is the thing to review in a diff.

## What the bank is not

- **Not a training corpus.** The `train` split exists for trainer review and
  gap analysis. The standing rule is that nothing auto-trains on evaluation
  items, and rubric items are never scored by the engine: a rubric skill stays
  unassessed until a trainer records a review.
- **Not held out from pretraining.** These datasets are public and almost
  certainly appear in the base model's pretraining data. `split: "eval"` means
  the item was never used for training inside Luke Lab; it does not mean the
  model has never seen it. Scores are reported with that caveat and must not be
  read as contamination-free benchmark numbers.
- **Not a licence-free bundle.** Each item names its dataset and licence. The
  CC BY-SA sources carry share-alike terms on the derived item text; the
  manifest is the attribution file to keep with any redistribution.
