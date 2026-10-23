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
- Use specific exceptions. No silent `except Exception: pass`.
- Do not import or add `numpy`.
- Add a test for every behavior change and bug fix.

## The brain's numbers

- Every number the app shows is measured from real files, models and events.
  Nothing invented, simulated, or placeholder — see `../CODE_STANDARDS.md`
  ("Honest measurements", strict rule).
- Any change that could move how well the model knows its lessons (training
  loop, writer, heads, request or correction path) is re-measured per
  `docs/07-model-measurements.md` and gets a row there in the same change.

## Dependencies and System Overview

When a package or runtime dependency changes, update the System Overview
package list in `assistance_front/src/app/tenants/tenants.component.ts` in the
same change. Include the package name and a plain-language description.

Development-only tools such as Ruff must be listed too.
