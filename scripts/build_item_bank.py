"""Build the vendored assessment item bank from its licensed public sources.

Each source is read at build time, shaped by the app's own intake code and
written into ``app/data/assessment/`` as one small JSON file per skill, with a
manifest that records the licence, the dataset, the row budget and the SHA-256
of every file written. Nothing here runs at request time and nothing it reads
is merged into the corpus the engine trains on.

Run from ``assistance_back/``:

    .venv/bin/python scripts/build_item_bank.py
    .venv/bin/python scripts/build_item_bank.py --check

``--check`` reads the bank back, verifies each digest against the manifest and
validates every stored item. It never touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "data" / "assessment"
MANIFEST = "manifest.json"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
FILE_URL = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{file}"
BANK_VERSION = 1
TIMEOUT_SECONDS = 60.0
ATTEMPTS = 6
RETRY_CODES = frozenset({429, 500, 502, 503, 504})
REQUEST_PAUSE = 1.5


class BuildError(Exception):
    """A source could not be read, or the bank it produced is not complete."""


def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "intake": importlib.import_module("app.services.assessment.intake"),
        "skills": importlib.import_module("app.services.assessment.skills"),
        "sources": importlib.import_module("app.services.assessment.sources"),
    }


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file, read in chunks so a large bank fits memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _http_get(url: str, params: dict[str, Any]) -> Any:
    """One GET, retried when the row server asks us to slow down."""
    import time

    import httpx

    response = httpx.get(url, params=params, timeout=TIMEOUT_SECONDS, follow_redirects=True)
    for attempt in range(ATTEMPTS - 1):
        if response.status_code not in RETRY_CODES:
            break
        time.sleep(min(2**attempt, 30))
        response = httpx.get(url, params=params, timeout=TIMEOUT_SECONDS, follow_redirects=True)
    return response


class Reader:
    """The row server, with one request per page and no repeats.

    Two skills read the same Dolly pages, so a page is kept for the run rather
    than fetched twice. The split's size comes from the first page, because the
    server reports it there and nowhere cheaper.
    """

    def __init__(self, get: Callable[[str, dict[str, Any]], Any]) -> None:
        self._get = get
        self._pages: dict[tuple[str, str, str, int], list[tuple[int, dict]]] = {}
        self._sizes: dict[tuple[str, str, str], int] = {}
        self._files: dict[tuple[str, str, str], list[tuple[int, dict]]] = {}

    def page(self, dataset: str, config: str, split: str, offset: int, length: int) -> list:
        key = (dataset, config, split, offset)
        if key not in self._pages:
            body = self._json(
                ROWS_URL,
                {
                    "dataset": dataset,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": length,
                },
            )
            self._sizes[(dataset, config, split)] = int(body.get("num_rows_total") or 0)
            self._pages[key] = [(int(row["row_idx"]), row["row"]) for row in body["rows"]]
        return self._pages[key]

    def total(self, dataset: str, config: str, split: str) -> int:
        key = (dataset, config, split)
        if key not in self._sizes:
            self.page(dataset, config, split, 0, 1)
        return self._sizes[key]

    def records(self, dataset: str, revision: str, path: str) -> list[tuple[int, dict]]:
        """A line-delimited file at a pinned revision, read one time."""
        key = (dataset, revision, path)
        if key not in self._files:
            self._pause()
            response = self._get(FILE_URL.format(dataset=dataset, revision=revision, file=path), {})
            if getattr(response, "status_code", 0) != 200:
                code = getattr(response, "status_code", "?")
                raise BuildError(f"{dataset}/{path} answered {code}")
            self._files[key] = _json_lines(response.text)
        return self._files[key]

    @staticmethod
    def _pause() -> None:
        import time

        time.sleep(REQUEST_PAUSE)

    def _json(self, url: str, params: dict[str, Any]) -> dict:
        self._pause()
        response = self._get(url, params)
        if getattr(response, "status_code", 0) != 200:
            code = getattr(response, "status_code", "?")
            raise BuildError(f"{url} answered {code} for {params.get('dataset')}")
        return json.loads(response.text)


def page_offsets(total: int, rows: int, page: int, spread: bool) -> list[int]:
    """The offsets to read, spaced across the split when the split is bigger.

    Spreading keeps a bank built from a long split from being one stretch of
    one subject, which would make a skill's score about a topic, not a skill.
    """
    if total <= 0 or not spread or total <= rows or total <= page:
        return [0] if total > 0 else []
    pages = max(2, min(-(-rows // page), (total - page) // page))
    last = total - page
    return [round(position * last / (pages - 1)) for position in range(pages)]


def _json_lines(text: str) -> list[tuple[int, dict]]:
    """Every JSON object in a line-delimited file, keyed by its line number."""
    rows: list[tuple[int, dict]] = []
    for number, line in enumerate(text.splitlines()):
        if line.strip():
            rows.append((number, json.loads(line)))
    return rows


def _pick_indices(total: int, rows: int, spread: bool) -> list[int]:
    """The positions to keep from a whole file, spaced across it when asked."""
    if total <= rows:
        return list(range(total))
    if not spread or rows < 2:
        return list(range(min(rows, total)))
    return sorted({round(position * (total - 1) / (rows - 1)) for position in range(rows)})


def _fetch_spec(reader: Reader, dataset: str, spec: Any, page: int) -> list[tuple[int, dict]]:
    """One fetch's rows, from its own file or through the paginated server."""
    if spec.file:
        everything = reader.records(dataset, spec.revision, spec.file)
        return [
            everything[index] for index in _pick_indices(len(everything), spec.rows, spec.spread)
        ]
    rows = list(reader.page(dataset, spec.config, spec.split, 0, page))
    offsets = page_offsets(
        reader.total(dataset, spec.config, spec.split), spec.rows, page, spec.spread
    )
    for offset in offsets[1:]:
        rows.extend(reader.page(dataset, spec.config, spec.split, offset, page))
    return rows[: spec.rows]


def fetch(reader: Reader, source: Any, page: int) -> list[tuple[int, dict]]:
    """Every row a source asks for, keyed by its position in the split."""
    records: list[tuple[int, dict]] = []
    for spec in source.fetches:
        records.extend(_fetch_spec(reader, source.meta.dataset, spec, page))
    return records


def _source_row(source: Any, measured: dict[str, int]) -> dict[str, Any]:
    meta = source.meta
    return {
        "key": meta.key,
        "skill": meta.skill,
        "dataset": meta.dataset,
        "licence": meta.licence,
        "url": meta.url,
        "records": meta.records,
        "rows": measured["rows"],
        "shaped": measured["shaped"],
        "fetches": [
            {
                "config": spec.config,
                "split": spec.split,
                "rows": spec.rows,
                "spread": spec.spread,
                "file": spec.file,
                "revision": spec.revision,
            }
            for spec in source.fetches
        ],
    }


def _file_row(skill: str, title: str, items: list[dict]) -> dict[str, Any]:
    path = TARGET / f"{skill}.json"
    return {
        "name": path.name,
        "skill": skill,
        "title": title,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "items": len(items),
        "eval": sum(1 for item in items if item["split"] == "eval"),
        "train": sum(1 for item in items if item["split"] == "train"),
        "kinds": sorted({item["expected"]["kind"] for item in items}),
        "stages": sorted({item["stage"] for item in items}),
        "topics": sorted({item["topic"] for item in items}),
    }


def _pool_by_skill(reader: Reader, app: dict[str, Any], page: int) -> tuple[dict, dict]:
    """The shaped items of every skill, and what it took to get them."""
    skills, registry = app["skills"], app["sources"]
    needed = skills.EVAL_ITEMS_PER_SKILL + skills.TRAIN_ITEMS_PER_SKILL
    pools: dict[str, list[dict]] = {}
    measured: dict[str, dict] = {}
    for source in registry.SOURCES.values():
        records = fetch(reader, source, page)
        items = registry.shape_pool(source, records)
        if len(items) < needed:
            raise BuildError(
                f"{source.meta.key} shaped {len(items)} items from {len(records)} rows; "
                f"a skill needs {needed}. Raise its row budget or add a source."
            )
        measured[source.meta.key] = {"rows": len(records), "shaped": len(items)}
        pools.setdefault(source.meta.skill, []).extend(items)
    return pools, measured


def build(reader: Reader) -> dict[str, Any]:
    """The bank, its files and the manifest that says how it was made."""
    app = _modules()
    intake, skills, registry = app["intake"], app["skills"], app["sources"]
    page = registry.PAGE_ROWS
    pools, measured = _pool_by_skill(reader, app, page)
    files = []
    for skill in sorted(pools):
        items = intake.assign_splits(pools[skill], registry.SPLIT_SALT)
        for item in items:
            intake.validate_item(item)
        if not any(item["split"] == "eval" for item in items):
            raise BuildError(f"the {skill} bank has no held-out items")
        title = skills.SKILL_TITLES[skill]
        _write_json(TARGET / f"{skill}.json", {"skill": skill, "title": title, "items": items})
        files.append(_file_row(skill, title, items))
    return {
        "bank": "luke-assessment-bank",
        "version": BANK_VERSION,
        "built_at": datetime.now(UTC).date().isoformat(),
        "salt": registry.SPLIT_SALT,
        "eval_items_per_skill": skills.EVAL_ITEMS_PER_SKILL,
        "train_items_per_skill": skills.TRAIN_ITEMS_PER_SKILL,
        "sources": [
            _source_row(source, measured[source.meta.key]) for source in registry.SOURCES.values()
        ],
        "files": files,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one file in one step, so a failed build leaves the old bank alone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _check_file(intake: Any, row: dict[str, Any]) -> None:
    path = TARGET / row["name"]
    if not path.is_file():
        raise BuildError(f"{path} is missing; run the builder without --check.")
    if sha256_file(path) != row["sha256"]:
        raise BuildError(f"{path} does not match the digest recorded in the manifest.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["skill"] != row["skill"] or payload["title"] != row["title"]:
        raise BuildError(f"{path} does not name the skill and title the manifest recorded.")
    if len(payload["items"]) != row["items"]:
        raise BuildError(
            f"{path} holds {len(payload['items'])} items, not the {row['items']} recorded."
        )
    for item in payload["items"]:
        intake.validate_item(item)


def check() -> dict[str, Any]:
    """Verify the bank on disk against its manifest, without the network."""
    app = _modules()
    intake, skills = app["intake"], app["skills"]
    recorded = json.loads((TARGET / MANIFEST).read_text(encoding="utf-8"))
    for row in recorded["files"]:
        _check_file(intake, row)
    missing = set(skills.SKILL_IDS) - {row["skill"] for row in recorded["files"]}
    if missing:
        raise BuildError(f"no bank file for {', '.join(sorted(missing))}")
    return recorded


def report(manifest: dict[str, Any], checked: bool) -> None:
    """What the bank holds, printed from the numbers the build just measured."""
    state = " (checked)" if checked else ""
    print(f"Assessment item bank: {TARGET}")
    print(f"  built   {manifest['built_at']} version {manifest['version']}{state}")
    print(f"  salt    {manifest['salt']}")
    for row in sorted(manifest["files"], key=lambda item: item["skill"]):
        print(
            f"  {row['skill']:17s} {row['items']:3d} items  "
            f"{row['eval']:2d} held out, {row['train']:2d} for training  "
            f"{row['bytes']:7d} bytes  kinds {', '.join(row['kinds'])}"
        )
    for row in manifest["sources"]:
        print(
            f"  source  {row['key']:14s} {row['dataset']:33s} "
            f"{row['licence']:12s} {row['rows']:5d} rows, {row['shaped']:4d} shaped"
        )


def main(argv: list[str] | None = None) -> int:
    """Build the bank, or check the bank that is already there."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without fetching")
    args = parser.parse_args(argv)
    try:
        manifest = check() if args.check else build(Reader(_http_get))
        if not args.check:
            _write_json(TARGET / MANIFEST, manifest)
    except (BuildError, OSError, KeyError, ValueError, TypeError) as error:
        print(f"Assessment item bank: {error}", file=sys.stderr)
        return 1
    report(manifest, args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
