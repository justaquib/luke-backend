# Single-user migration (Luke Lab)

Luke Lab serves one workspace. The canonical workspace is **0**; the
`tenant_id` columns stay in the schema, defaulted to 0, so a future multi-user
path does not need a second migration of the data shape. The Python package,
the database name (`assistance`) and the file layout are unchanged.

## What the migration does

1. Backs up the database (a JSON dump of every table), `.chroma`,
   `.language_model` and `uploads` into `backups/single-user-<timestamp>/`,
   with a size and SHA-256 per file.
2. Inserts workspace 0 (`Luke Lab`) if it is missing.
3. Remaps every row of `documents`, `uploaded_files`, `faqs`, `feedback`,
   `pending_questions`, `intents`, `users`, `invites`, `rules`,
   `website_crawls` and `website_pages` onto workspace 0.
4. Merges FAQs that ask the same question (casefolded, whitespace-collapsed):
   the approved or latest answer is kept, `intents.faq_id` pointers move to the
   kept row, and every dropped row is recorded in the undo file.
5. Keeps the oldest system administrator and removes the other users and
   tenants. Invites created by a removed user move to the kept one.
6. Merges `tenant-*.json` learner state into `tenant-0.json` by the SHA-256 of
   the question and answer; a conflicting id is reassigned. Files measured
   against another model (`instruction-*.json`) are kept as history.
7. Labels every vector with `tenant_id: "0"` in the `docs`, `faq`, `intents`
   and `lessons` collections.
8. Moves loose uploads into `uploads/0`. A same-name, different-content file is
   reported as a conflict and left in place.
9. Verifies counts, zero rows outside workspace 0, zero duplicate FAQ groups,
   zero unlabelled vectors, zero loose uploads, and one fixed retrieval query.

No column is dropped. The migration is idempotent: a second run reports
`0 to remap`, `0 to label` and `0` loose uploads.

## Commands

Run from `assistance_back/` with the API stopped:

```bash
.venv/bin/python scripts/migrate_single_user.py --dry-run
.venv/bin/python scripts/migrate_single_user.py --yes
```

Both write `report.json` and `conflicts.json` under
`backups/preflight-<timestamp>/`. `--yes` also writes the backup, `undo.json`
and a `rollback.md` naming the restore command.

Restore the database, and optionally the files, from a backup:

```bash
.venv/bin/python scripts/restore_single_user.py backups/single-user-<timestamp> --yes

# rehearse on a copy: a scratch database and a temporary files directory
.venv/bin/python scripts/restore_single_user.py backups/single-user-<timestamp> --yes \
  --database-url postgresql+psycopg2://postgres:postgres@localhost:5433/luke_restore_test \
  --files-target /tmp/luke-restore-test
```

`--database-url` copies the schema into that database and restores the rows;
point it at a scratch database first. `--files-target` restores `.chroma`,
`.language_model` and `uploads` there instead of over the live folders. Without
`--database-url` the live database is restored in place, so stop the API and
take a fresh backup first.

The backup is a JSON dump rather than `pg_dump` output because the installed
client tools are PostgreSQL 14 and the server is 16; `pg_dump` refuses to read
a newer server. `restore_tables` recreates the schema, inserts parents before
children and resets each sequence.

## Applied run (2026-10-27)

Backup: `backups/single-user-20261027-162811` (database.json 652,256 bytes;
`.chroma` 279,956 KiB; `.language_model` 279,624 KiB; `uploads` 14,552 KiB).

| Check | Before | After |
| --- | --- | --- |
| Rows outside workspace 0 | 474 | 0 |
| Duplicate FAQ groups | 0 | 0 |
| Vectors without the workspace | 1,066 | 0 |
| Loose uploads | 4 | 0 |
| Users / tenants | 2 / 2 | 1 / 1 |

The fixed retrieval query ("What is there to know about wind?") returned
nothing before the run because the vectors carried no workspace metadata, and
returns the document and FAQ hits after it.

The rollback was rehearsed on a scratch database (`luke_rollback_test`) and a
temporary files directory: restored counts matched `manifest.json` exactly, and
the restored copy showed the pre-migration shape (tenant 1, NULL document
buckets, two users). The scratch database and files were then removed.

## Code changes that support the single workspace

- `POST /ingest` passes the workspace to `ingest_text` and stores it on the
  document and its vectors.
- `/documents/{filename}/view` and `/download` require a session and resolve
  only inside the caller's workspace.
- Uploads are validated (extension allowlist, MIME sniff, size cap) and
  deduplicated by content hash; a same-name different-content upload is a 409.
- Chunk ids are `uuid5(namespace, "<workspace>:<source>:<sha256(text)>")` and
  written with `upsert`, so re-indexing overwrites.
- The reported similarity reads the collection's distance space (`1 - d/2` for
  L2 squared, `1 - d` otherwise).
- Startup resync writes the workspace id and only touches changed rows.
- FAQ and intent ownership checks compare against `current_workspace()`; no
  service reads the signed-in user's `tenant_id` to choose a workspace.
- A new Google sign-in joins workspace 0 and reuses its existing tenant row;
  no tenant row is created per person.
- `/auth/tenants` is removed, together with the tenant administration screen
  and its client calls. Invitation routes remain part of the login flow.
