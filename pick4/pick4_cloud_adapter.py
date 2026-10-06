"""Cloud source adapter for the locked Pick 4 model.

The model file remains byte-for-byte locked. This adapter changes only the
secondary archive transport when sc.pick-4.com is unreachable from a runner.
"""
import hashlib
import json
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

import pick4_track as locked


used_fallback = False
original_acquire_history = locked.acquire_history


def parse_lottery_net(body, draw_type):
    soup = BeautifulSoup(body, "html.parser")
    records = []
    for row in soup.select("tr"):
        date_node = row.select_one("td.pickArchive")
        if not date_node:
            continue
        try:
            date = datetime.strptime(date_node.get_text(" ", strip=True), "%A %B %d, %Y").date().isoformat()
        except ValueError:
            continue
        digits = [node.get_text(strip=True) for node in row.select('li[class*="number-part-"]:not(.fireball)')]
        if len(digits) == 4 and all(len(value) == 1 and value.isdigit() for value in digits):
            records.append({"date": date, "draw_type": draw_type, "number": "".join(digits)})
    if not records:
        raise ValueError("Lottery.net Pick 4 fallback did not parse")
    return records


def fallback_history():
    global used_fallback
    try:
        return original_acquire_history()
    except Exception as primary_error:
        used_fallback = True
        raw = locked.OUT / "sources" / "history"
        current_year = locked.utcnow().astimezone(locked.TZ).year
        records = []
        snapshots = {}
        for year in range(2003, current_year):
            canonical = raw / f"archive_{year}.html"
            body = canonical.read_bytes()
            records.extend(locked.parse_archive(body))
            snapshots[str(year)] = {"file": str(canonical.relative_to(locked.OUT)), "sha256": locked.digest(canonical)}
        current_snapshots = []
        for draw_type, slug in (("Day", "pick-4-midday"), ("Night", "pick-4-evening")):
            body = locked.public_page(f"https://www.lottery.net/south-carolina/{slug}/numbers/{current_year}")
            stream = parse_lottery_net(body, draw_type)
            records.extend(stream)
            state = hashlib.sha256(json.dumps(stream, sort_keys=True).encode()).hexdigest()[:20]
            path = raw / f"lotterynet_{current_year}_{draw_type}_{state}.html"
            if not path.exists():
                path.write_bytes(body)
            current_snapshots.append({"file": str(path.relative_to(locked.OUT)), "sha256": locked.digest(path)})
        snapshots[str(current_year)] = {
            "provider": "lottery.net fallback",
            "files": current_snapshots,
            "primary_error": str(primary_error)[:240],
        }
        frame = pd.DataFrame(records).drop_duplicates(["date", "draw_type"], keep="last").sort_values(["date", "draw_type"])
        if frame.duplicated(["date", "draw_type"]).any() or not frame.number.str.fullmatch(r"\d{4}").all():
            raise ValueError("Invalid Pick 4 fallback history")
        return frame.reset_index(drop=True), snapshots


locked.acquire_history = fallback_history
locked.main()

if used_fallback:
    status_path = locked.OUT / "source_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["archive_provider"] = "lottery.net fallback; official SC Lottery agreement still required"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
