"""Re-read the allowlisted web sources that are due, and drop expired pages.

This is the entry point a scheduler calls. It reads the workspace policy from
``.language_model``, re-reads only the sources whose snapshot has passed its
time to live, keeps an unchanged page as it is, and records a changed page as a
new snapshot. Nothing is trained from a page; a general page still waits in the
trainer's inbox.

Run it by hand with:
    .venv/bin/python scripts/refresh_web.py

A daily schedule is enough for a weekly time to live:
    crontab -e
    17 3 * * * cd /path/to/assistance_back && .venv/bin/python scripts/refresh_web.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    """Refresh one workspace and print the measured report."""
    from app.db.database import SessionLocal
    from app.services.web import ingest, trust

    with SessionLocal() as db:
        policy = trust.read(None)
        if not policy.enabled:
            print("Web evidence is off; nothing to refresh.")
            return 0
        report = ingest.run_refresh(db, policy, None)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
