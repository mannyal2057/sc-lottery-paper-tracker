"""Run cloud paper-study updates without losing the public dashboard to a source outage."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).parent
STATUS_PATH = ROOT / "cloud_status.json"


def run(command: list[str], cwd: Path, attempts: int = 1) -> dict:
    last = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        last = result
        if result.returncode == 0:
            return {"ok": True, "attempts": attempt, "message": "updated"}
        if attempt < attempts:
            time.sleep(10 * attempt)

    output = (last.stdout or "").strip().splitlines()
    return {
        "ok": False,
        "attempts": attempts,
        "message": " | ".join(output[-3:])[:600] or "update command failed",
    }


def main() -> None:
    py = ["python"]
    pick3_dir = ROOT / "pick3"
    pick4_dir = ROOT / "pick4"

    status = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "pick3_source": run(py + ["paper_track.py"], pick3_dir, attempts=3),
        "pick3_supporting": {},
        "pick4_source": run(py + ["pick4_cloud_adapter.py"], pick4_dir, attempts=3),
    }

    # These scripts use the currently verified records. If a source is temporarily
    # unavailable they retain pending outcomes rather than inventing a result.
    for script in (
        "three_pick_review.py",
        "weather_experiment.py",
        "equipment_context.py",
        "box_track.py",
        "challenger_cloud_adapter.py",
        "cdm_cloud_adapter.py",
    ):
        status["pick3_supporting"][script] = run(py + [script], pick3_dir)

    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    for label in ("pick3_source", "pick4_source"):
        state = status[label]
        print(f"{label}: {'ok' if state['ok'] else 'unavailable'} after {state['attempts']} attempt(s)")
    failed_supporting = [name for name, state in status["pick3_supporting"].items() if not state["ok"]]
    if failed_supporting:
        print("supporting update failures: " + ", ".join(failed_supporting))


if __name__ == "__main__":
    main()
