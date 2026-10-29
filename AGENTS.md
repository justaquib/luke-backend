# Backend agent instructions

## Mandatory before any implementation

1. Read `../CODE_STANDARDS.md`.
2. Read `../CODE_QUALITY_PLAN.md`.
3. Run `make quality` before reporting work complete.
4. Never disable, weaken, or bypass a rule without explicit approval and a
   documented exception in `CODE_QUALITY_PLAN.md`.

## Backend structure

- `app/main.py` only creates the app, registers middleware, and includes
  routers.
- HTTP routes live in `app/routers/` and stay thin. Business logic lives in
  `app/services/`.
- Routers call services; services do not import routers.
- Database access stays out of routers and pure business logic.
- Request and response data uses Pydantic schemas.
- Use specific exceptions. No silent `except Exception: pass`. A broad catch at
  a boundary logs the failure it absorbs and carries
  `# noqa: BLE001 -- <boundary>` on the `except` line; `BLE001` is in the gate.
  See `../CODE_STANDARDS.md`.
- Do not import or add `numpy`.
- Add a test for every behavior change and bug fix.

## The brain's numbers

- Every number the app shows is measured from real files, models and events.
  Nothing invented, simulated, or placeholder — see `../CODE_STANDARDS.md`
  ("Honest measurements", strict rule).
- Any change that could move how well the model knows its lessons (training
  loop, word model, heads, request or correction path) is re-measured per
  `docs/07-model-measurements.md` and gets a row there in the same change.

## The neural engine

- The only served model is the word model (`app/services/neural/words.py`):
  next-word prediction over the tenant's sentences, used by the learner's
  sentence feature. It has no part in writing a chat answer.
- The chat may answer from the trainer's material: a taught lesson, an
  approved answer, a document, a dictionary entry or a conversation pair. A
  grounded answer must cite the evidence it used and pass the verifier.
- A generated answer is allowed, must be labelled as generated, and must not
  fabricate a citation, an identity or a measurement. Retrieval is optional
  evidence, not a gate. (This replaces the earlier "no model writes an answer"
  rule; see `../CODE_QUALITY_PLAN.md`.)
- The checkpoint is written in its own process and reloaded when the file
  changes. The boundary catch in `brain_training` logs the failure and records
  the stop reason, so a failed run is reported rather than swallowed.

## Dependencies and System Overview

When a package or runtime dependency changes, update the System Overview
package list in `assistance_front/src/app/settings/system-overview.data.ts` in
the same change. Include the package name and a plain-language description.

Development-only tools such as Ruff must be listed too.
