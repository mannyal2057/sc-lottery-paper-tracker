"""Locked prospective SC Pick 4 study using public, no-login sources."""
import hashlib
import itertools
import json
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).parent
OUT = ROOT / "study"
TZ = ZoneInfo("America/New_York")
VERSION = "SC-PICK4-CHALLENGER-V1"
SEED = 20260926
SIGNAL_WEIGHT = 0.15
COMPONENT_WEIGHTS = {"position": 0.35, "pairs": 0.20, "transition": 0.15, "repeat_gap": 0.10, "change_point": 0.20}
DIGITS = np.array([[n // 1000, (n // 100) % 10, (n // 10) % 10, n % 10] for n in range(10000)], dtype=int)
BOX_PRIZES = {4: 1200, 6: 800, 12: 400, 24: 200}
BOX_GROUPS = {}
for number in range(10000):
    BOX_GROUPS.setdefault("".join(sorted(f"{number:04d}")), []).append(number)
BOX_GROUPS = {key: values for key, values in BOX_GROUPS.items() if len(values) in BOX_PRIZES}


def utcnow():
    return datetime.now(timezone.utc)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def public_page(url):
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(
        url + separator + "paper_check=" + str(int(utcnow().timestamp())),
        headers={"User-Agent": "Mozilla/5.0 SC-Pick4-paper-study", "Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def deadline(date, draw_type):
    time = "T12:45:00" if draw_type == "Day" else "T18:45:00"
    return datetime.fromisoformat(date + time).replace(tzinfo=TZ)


def next_day(date, draw_type):
    day = datetime.fromisoformat(date).date() + timedelta(days=1)
    while draw_type == "Day" and (day.weekday() == 6 or (day.month, day.day) == (12, 25)):
        day += timedelta(days=1)
    return day.isoformat()


def parse_archive(body):
    soup = BeautifulSoup(body, "html.parser")
    records = []
    for container in soup.select("div.drawContainer"):
        date_node = container.select_one("a.date span")
        if not date_node:
            continue
        date = datetime.strptime(date_node.get_text(" ", strip=True), "%A, %B %d, %Y").date().isoformat()
        for balls in container.select("ul.drawBalls"):
            if balls.select_one(".middayDraw"):
                draw_type = "Day"
            elif balls.select_one(".eveningDraw"):
                draw_type = "Night"
            else:
                continue
            digits = [node.get_text(strip=True) for node in balls.select("li[class*='number-part-']")]
            if len(digits) != 4 or not all(value.isdigit() and len(value) == 1 for value in digits):
                raise ValueError("Archive Pick 4 row did not parse")
            records.append({"date": date, "draw_type": draw_type, "number": "".join(digits)})
    if not records:
        raise ValueError("Archive source did not parse")
    return records


def parse_official(body):
    soup = BeautifulSoup(body, "html.parser")
    results = {}
    for box in soup.select(".drawResultsaccordion-title"):
        date_node = box.select_one(".lightblue-bg")
        kind_node = box.select_one(".lightblue-bg-right")
        if not date_node or not kind_node:
            continue
        date = datetime.strptime(date_node.get_text(strip=True), "%B %d, %Y").date().isoformat()
        label = kind_node.get_text(strip=True)
        if label not in {"Midday", "Evening"}:
            continue
        number = "".join(node.get_text(strip=True) for node in box.select("li.number:not(.fireball)"))
        if len(number) != 4 or not number.isdigit():
            raise ValueError("Official Pick 4 row did not parse")
        results[(date, "Day" if label == "Midday" else "Night")] = number
    if not results:
        raise ValueError("Official source did not parse")
    return results


def acquire_history():
    raw = OUT / "sources" / "history"
    raw.mkdir(parents=True, exist_ok=True)
    current_year = utcnow().astimezone(TZ).year
    records = []
    snapshots = {}
    for year in range(2003, current_year + 1):
        canonical = raw / f"archive_{year}.html"
        if year == current_year or not canonical.exists():
            body = public_page(f"https://sc.pick-4.com/winning-numbers/{year}")
            parsed = parse_archive(body)
            if year < current_year:
                canonical.write_bytes(body)
            else:
                state = hashlib.sha256(json.dumps(parsed, sort_keys=True).encode()).hexdigest()[:20]
                canonical = raw / f"archive_{year}_{state}.html"
                if not canonical.exists():
                    canonical.write_bytes(body)
        else:
            body = canonical.read_bytes()
            parsed = parse_archive(body)
        records.extend(parsed)
        snapshots[str(year)] = {"file": str(canonical.relative_to(OUT)), "sha256": digest(canonical)}
    frame = pd.DataFrame(records).drop_duplicates(["date", "draw_type"], keep="last").sort_values(["date", "draw_type"])
    if frame.duplicated(["date", "draw_type"]).any() or not frame.number.str.fullmatch(r"\d{4}").all():
        raise ValueError("Invalid Pick 4 history")
    return frame.reset_index(drop=True), snapshots


def fetch_current():
    source_folder = OUT / "sources"
    body = public_page("https://www.sceducationlottery.com/Games/Pick4")
    official = parse_official(body)
    state = hashlib.sha256(json.dumps(sorted((d, k, n) for (d, k), n in official.items())).encode()).hexdigest()[:20]
    path = source_folder / f"official_{state}.html"
    if not path.exists():
        path.write_bytes(body)
    return official, path


def normalize(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (10000,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Invalid Pick 4 probability component")
    return values / values.sum()


def position_component(draws):
    components = []
    for window, weight in ((50, 0.50), (200, 0.30), (None, 0.20)):
        sample = draws[-window:] if window else draws
        probs = np.empty((4, 10))
        for position in range(4):
            counts = np.bincount(sample[:, position], minlength=10).astype(float) + 20.0
            probs[position] = counts / counts.sum()
        components.append((weight, normalize(np.prod(probs[np.arange(4)[:, None], DIGITS.T], axis=0))))
    return normalize(sum(weight * probability for weight, probability in components))


def pair_component(draws):
    sample = draws[-500:]
    score = np.ones(10000)
    for left, right in itertools.combinations(range(4), 2):
        counts = np.full((10, 10), 5.0)
        np.add.at(counts, (sample[:, left], sample[:, right]), 1.0)
        counts /= counts.sum()
        score *= counts[DIGITS[:, left], DIGITS[:, right]]
    return normalize(score)


def transition_component(draws):
    sample = draws[-1001:]
    previous = sample[-1]
    score = np.ones(10000)
    for position in range(4):
        counts = np.full((10, 10), 5.0)
        np.add.at(counts, (sample[:-1, position], sample[1:, position]), 1.0)
        row = counts[previous[position]] / counts[previous[position]].sum()
        score *= row[DIGITS[:, position]]
    return normalize(score)


def gap_bucket(gap):
    return gap if gap <= 4 else (5 if gap <= 9 else 6)


def repeat_gap_component(draws):
    sample = draws[-2000:]
    probs = np.empty((4, 10))
    for position in range(4):
        exposures = np.zeros((10, 7))
        events = np.zeros((10, 7))
        last_seen = np.full(10, -1, dtype=int)
        for index, observed in enumerate(sample[:, position]):
            if index:
                for digit in range(10):
                    gap = index - last_seen[digit] - 1 if last_seen[digit] >= 0 else 10
                    bucket = gap_bucket(int(gap))
                    exposures[digit, bucket] += 1
                    if observed == digit:
                        events[digit, bucket] += 1
            last_seen[observed] = index
        hazards = np.empty(10)
        for digit in range(10):
            gap = len(sample) - last_seen[digit] - 1 if last_seen[digit] >= 0 else 10
            bucket = gap_bucket(int(gap))
            hazards[digit] = (events[digit, bucket] + 2.0) / (exposures[digit, bucket] + 20.0)
        probs[position] = hazards / hazards.sum()
    return normalize(np.prod(probs[np.arange(4)[:, None], DIGITS.T], axis=0))


def change_point_component(draws):
    recent = draws[-50:]
    reference = draws[-250:-50] if len(draws) >= 250 else draws[:-50]
    if len(reference) < 20:
        reference = draws
    probs = np.empty((4, 10))
    for position in range(4):
        recent_counts = np.bincount(recent[:, position], minlength=10).astype(float) + 10.0
        reference_counts = np.bincount(reference[:, position], minlength=10).astype(float) + 20.0
        recent_rate = recent_counts / recent_counts.sum()
        reference_rate = reference_counts / reference_counts.sum()
        shift = np.clip(np.log(recent_rate / reference_rate), -0.35, 0.35)
        adjusted = reference_rate * np.exp(shift)
        probs[position] = adjusted / adjusted.sum()
    return normalize(np.prod(probs[np.arange(4)[:, None], DIGITS.T], axis=0))


def probability_model(numbers):
    draws = np.asarray(numbers, dtype=int)
    if draws.ndim != 2 or draws.shape[1] != 4 or len(draws) < 250 or np.any(draws < 0) or np.any(draws > 9):
        raise ValueError("Pick 4 model requires at least 250 valid preceding same-stream draws")
    components = {
        "position": position_component(draws),
        "pairs": pair_component(draws),
        "transition": transition_component(draws),
        "repeat_gap": repeat_gap_component(draws),
        "change_point": change_point_component(draws),
    }
    signal = normalize(sum(COMPONENT_WEIGHTS[name] * components[name] for name in COMPONENT_WEIGHTS))
    return normalize((1 - SIGNAL_WEIGHT) / 10000 + SIGNAL_WEIGHT * signal), components


def top_numbers(probabilities):
    tie = np.random.default_rng(SEED).random(10000)
    order = np.lexsort((tie, -np.asarray(probabilities)))
    return [f"{number:04d}" for number in order[:3]]


def select_boxes(probabilities):
    p = np.asarray(probabilities)
    ranked = sorted(BOX_GROUPS, key=lambda key: (-float(p[BOX_GROUPS[key]].sum()), key))[:3]
    return [{"digits": key, "ways": len(BOX_GROUPS[key]), "probability": float(p[BOX_GROUPS[key]].sum()), "gross_prize_usd": BOX_PRIZES[len(BOX_GROUPS[key])]} for key in ranked]


def controls(date, draw_type, boxes):
    seed = int(hashlib.sha256(f"{VERSION}|{date}|{draw_type}|{SEED}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    straights = [f"{number:04d}" for number in rng.choice(10000, 3, replace=False)]
    random_boxes = []
    used = set()
    for box in boxes:
        eligible = [key for key, values in BOX_GROUPS.items() if len(values) == box["ways"] and key not in used]
        key = str(rng.choice(eligible))
        used.add(key)
        random_boxes.append({"digits": key, "ways": box["ways"], "gross_prize_usd": BOX_PRIZES[box["ways"]]})
    return straights, random_boxes


def box_payout(boxes, actual):
    key = "".join(sorted(actual))
    return sum(box["gross_prize_usd"] for box in boxes if box["digits"] == key)


def append_audit(path):
    audit = OUT / "audit.jsonl"
    previous = "0" * 64
    known = set()
    if audit.exists():
        for line in audit.read_text().splitlines():
            row = json.loads(line)
            chain = row.pop("chain_hash")
            if row["previous"] != previous or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() != chain:
                raise ValueError("Pick 4 audit chain mismatch")
            if digest(OUT / row["file"]) != row["sha256"]:
                raise ValueError("Pick 4 registered record changed")
            previous = chain
            known.add(row["file"])
    relative = path.relative_to(OUT).as_posix()
    if relative in known:
        return
    row = {"file": relative, "sha256": digest(path), "previous": previous, "logged_utc": utcnow().isoformat()}
    row["chain_hash"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def initialize(history, snapshots):
    for name in ("forecasts", "results", "sources"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["code_sha256"] != digest(Path(__file__)):
            raise ValueError("Locked Pick 4 code changed; create a new prospective version")
        return protocol
    history_bytes = history.to_csv(index=False).encode()
    frozen = OUT / "history_initial.csv"
    frozen.write_bytes(history_bytes)
    protocol = {
        "version": VERSION,
        "status": "LOCKED_PROSPECTIVE",
        "created_utc": utcnow().isoformat(),
        "code_sha256": digest(Path(__file__)),
        "initial_history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "initial_source_snapshots": snapshots,
        "model": {"uniform_weight": 0.85, "signal_weight": SIGNAL_WEIGHT, "component_weights": COMPONENT_WEIGHTS, "features": "Same-stream position frequencies, all six position pairs, previous-draw transitions, repeat-gap hazards and bounded recent change."},
        "registration": "Before 12:45 Day / 18:45 Night America/New_York; verified preceding history only; exclusive-create; never backfill, overwrite or retune.",
        "selections": "Three straights and three non-triple Box classes. Seeded random controls match straight count and Box permutation composition exactly.",
        "paper_wagers": {"straight": "$1 each; $3 per draw; $5,000 gross exact prize", "box": "$1 each; $3 per draw; gross prizes 4-way $1,200, 6-way $800, 12-way $400, 24-way $200", "fireball": "excluded"},
        "sources": {"official": "https://www.sceducationlottery.com/Games/Pick4", "archive": "https://sc.pick-4.com/winning-numbers/{year}", "rules": "https://sceducationlottery.com/documents/games/onlinegames/GameRules_Pick4.pdf"},
        "evaluation": "Score only when official and archive agree. Report straight/Box hits, matched controls, net and log loss versus uniform. First descriptive Box checkpoint at 100 scored draws; final at 200 scored draws per stream.",
        "limitations": "Paper experiment only. Historical patterns may have no predictive value; no tickets purchased and no advantage claimed from interim results.",
        "seed": SEED,
    }
    write_once(protocol_path, protocol)
    return protocol


def register(history, protocol, official):
    now = utcnow()
    for draw_type in ("Day", "Night"):
        stream = history[history.draw_type == draw_type].sort_values("date")
        last = stream.date.max()
        target = next_day(last, draw_type)
        path = OUT / "forecasts" / f"{target}_{draw_type}.json"
        if path.exists() or now >= deadline(target, draw_type):
            continue
        if (last, draw_type) not in official or official[(last, draw_type)] != stream.iloc[-1].number:
            continue
        numbers = np.array([[int(value) for value in number] for number in stream.number], dtype=int)
        probabilities, components = probability_model(numbers)
        boxes = select_boxes(probabilities)
        random_straights, random_boxes = controls(target, draw_type, boxes)
        registered = utcnow()
        if registered >= deadline(target, draw_type):
            continue
        history_snapshot = json.dumps(stream[["date", "number"]].to_dict("records"), sort_keys=True, separators=(",", ":")).encode()
        record = {
            "version": VERSION, "date": target, "draw_type": draw_type, "registered_utc": registered.isoformat(),
            "deadline_local": deadline(target, draw_type).isoformat(), "training_through": last,
            "protocol_sha256": digest(OUT / "protocol.json"), "history_sha256": hashlib.sha256(history_snapshot).hexdigest(),
            "probabilities": probabilities.tolist(), "component_top3": {name: top_numbers(value) for name, value in components.items()},
            "straight_picks": top_numbers(probabilities), "random_straight_picks": random_straights,
            "box_picks": boxes, "random_box_picks": random_boxes, "straight_cost_usd": 3, "box_cost_usd": 3,
        }
        write_once(path, record)
        append_audit(path)


def score(history, official, official_path):
    archive = {(row.date, row.draw_type): row.number for row in history.itertuples()}
    pending = []
    for forecast_path in sorted((OUT / "forecasts").glob("*.json")):
        forecast = json.loads(forecast_path.read_text())
        key = (forecast["date"], forecast["draw_type"])
        if key in official and key not in archive:
            pending.append({"date": key[0], "draw_type": key[1], "official_number": official[key], "status": "Awaiting archive confirmation; not scored"})
        result_path = OUT / "results" / forecast_path.name
        if result_path.exists() or key not in official or key not in archive:
            continue
        if official[key] != archive[key]:
            raise ValueError("Pick 4 sources disagree at " + str(key))
        actual = official[key]
        model_box_gross = box_payout(forecast["box_picks"], actual)
        random_box_gross = box_payout(forecast["random_box_picks"], actual)
        result = {
            "date": key[0], "draw_type": key[1], "actual": actual, "scored_utc": utcnow().isoformat(),
            "forecast_sha256": digest(forecast_path), "official_snapshot": str(official_path.relative_to(OUT)),
            "official_snapshot_sha256": digest(official_path), "straight_hit": actual in forecast["straight_picks"],
            "random_straight_hit": actual in forecast["random_straight_picks"], "box_hit": model_box_gross > 0,
            "random_box_hit": random_box_gross > 0, "model_log_loss": float(-np.log(forecast["probabilities"][int(actual)])),
            "uniform_log_loss": float(np.log(10000.0)), "straight_gross_usd": 5000 if actual in forecast["straight_picks"] else 0,
            "straight_net_usd": (5000 if actual in forecast["straight_picks"] else 0) - 3,
            "random_straight_net_usd": (5000 if actual in forecast["random_straight_picks"] else 0) - 3,
            "box_gross_usd": model_box_gross, "box_net_usd": model_box_gross - 3,
            "random_box_gross_usd": random_box_gross, "random_box_net_usd": random_box_gross - 3,
            "best_position_matches": max(sum(a == b for a, b in zip(pick, actual)) for pick in forecast["straight_picks"]),
            "best_digit_overlap": max(sum((Counter(pick) & Counter(actual)).values()) for pick in forecast["straight_picks"]),
        }
        write_once(result_path, result)
        append_audit(result_path)
    (OUT / "source_status.json").write_text(json.dumps({"checked_utc": utcnow().isoformat(), "awaiting_archive_confirmation": pending}, indent=2), encoding="utf-8")


def render(protocol):
    rows = []
    for forecast_path in sorted((OUT / "forecasts").glob("*.json")):
        forecast = json.loads(forecast_path.read_text())
        result_path = OUT / "results" / forecast_path.name
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        rows.append({"date": forecast["date"], "draw_type": forecast["draw_type"], "registered_utc": forecast["registered_utc"], "straight_picks": forecast["straight_picks"], "box_picks": forecast["box_picks"], "result": result})
    scored = [row["result"] for row in rows if row["result"]]
    totals = {
        "scored_draws": len(scored), "straight_hits": sum(result["straight_hit"] for result in scored),
        "random_straight_hits": sum(result["random_straight_hit"] for result in scored), "box_hits": sum(result["box_hit"] for result in scored),
        "random_box_hits": sum(result["random_box_hit"] for result in scored), "straight_net_usd": sum(result["straight_net_usd"] for result in scored),
        "random_straight_net_usd": sum(result["random_straight_net_usd"] for result in scored), "box_net_usd": sum(result["box_net_usd"] for result in scored),
        "random_box_net_usd": sum(result["random_box_net_usd"] for result in scored),
        "mean_model_log_loss": float(np.mean([result["model_log_loss"] for result in scored])) if scored else None,
        "mean_uniform_log_loss": float(np.mean([result["uniform_log_loss"] for result in scored])) if scored else None,
    }
    summary = {"version": VERSION, "status": protocol["status"], "updated_utc": utcnow().isoformat(), "protocol_sha256": digest(OUT / "protocol.json"), "totals": totals, "checkpoint": {"first_box_checkpoint_draws": 100, "final_draws_per_stream": 200}, "rows": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = "# SC Pick 4 — locked prospective study\n\nThree $1 paper straights and three $1 paper Boxes per drawing. FIREBALL excluded. No ticket purchases.\n\n"
    text += "| Date | Drawing | Three straights | Three boxes | Actual | Straight | Box |\n|---|---|---|---|---|---|---|\n"
    for row in rows:
        result = row["result"]
        boxes = ", ".join(f"{box['digits']} ({box['ways']}-way)" for box in row["box_picks"])
        text += f"| {row['date']} | {row['draw_type']} | {', '.join(row['straight_picks'])} | {boxes} | {result['actual'] if result else 'Pending'} | {result['straight_hit'] if result else 'Pending'} | {result['box_hit'] if result else 'Pending'} |\n"
    text += "\nThe model is 85% uniform and 15% fixed historical signal. Wider Box coverage is not a predictive advantage. Compare every result with the composition-matched random control.\n"
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    for name in ("forecasts", "results", "sources"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    history, snapshots = acquire_history()
    official, official_path = fetch_current()
    for key, number in official.items():
        match = history[(history.date == key[0]) & (history.draw_type == key[1])]
        if len(match) and match.iloc[0].number != number:
            raise ValueError("Official and archive Pick 4 sources disagree at " + str(key))
    protocol = initialize(history, snapshots)
    score(history, official, official_path)
    register(history, protocol, official)
    render(protocol)


if __name__ == "__main__":
    main()
