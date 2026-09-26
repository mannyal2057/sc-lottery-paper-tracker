"""Locked prospective CHALLENGER-V2 for SC Pick 3.

The challenger runs beside the frozen baseline. It never rewrites or backfills a
forecast and scores only results already verified by the baseline study.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paper_track import PAPER, ROOT, deadline, digest, write_once


OUT = ROOT / "challenger_v2"
VERSION = "CHALLENGER-V2"
SEED = 20260926
SIGNAL_WEIGHT = 0.15
COMPONENT_WEIGHTS = {
    "position": 0.35,
    "pairs": 0.20,
    "transition": 0.15,
    "repeat_gap": 0.10,
    "change_point": 0.20,
}
DIGITS = np.array([[n // 100, (n // 10) % 10, n % 10] for n in range(1000)], dtype=int)
BOX_GROUPS = {}
for number in range(1000):
    BOX_GROUPS.setdefault("".join(sorted(f"{number:03d}")), []).append(number)
BOX_GROUPS = {key: values for key, values in BOX_GROUPS.items() if len(values) in (3, 6)}


def utcnow():
    return datetime.now(timezone.utc)


def normalize(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (1000,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid probability component")
    total = values.sum()
    if total <= 0:
        raise ValueError("Empty probability component")
    return values / total


def position_component(draws):
    distributions = []
    for window, weight in ((50, 0.50), (200, 0.30), (None, 0.20)):
        sample = draws[-window:] if window else draws
        probs = np.empty((3, 10), dtype=float)
        for position in range(3):
            counts = np.bincount(sample[:, position], minlength=10).astype(float) + 20.0
            probs[position] = counts / counts.sum()
        distributions.append((weight, normalize(np.prod(probs[np.arange(3)[:, None], DIGITS.T], axis=0))))
    return normalize(sum(weight * probability for weight, probability in distributions))


def pair_component(draws):
    sample = draws[-500:]
    score = np.ones(1000, dtype=float)
    for left, right in ((0, 1), (1, 2), (0, 2)):
        counts = np.full((10, 10), 5.0, dtype=float)
        np.add.at(counts, (sample[:, left], sample[:, right]), 1.0)
        counts /= counts.sum()
        score *= counts[DIGITS[:, left], DIGITS[:, right]]
    return normalize(score)


def transition_component(draws):
    sample = draws[-1001:]
    score = np.ones(1000, dtype=float)
    previous = sample[-1]
    for position in range(3):
        counts = np.full((10, 10), 5.0, dtype=float)
        if len(sample) > 1:
            np.add.at(counts, (sample[:-1, position], sample[1:, position]), 1.0)
        row = counts[previous[position]]
        row /= row.sum()
        score *= row[DIGITS[:, position]]
    return normalize(score)


def gap_bucket(gap):
    if gap <= 4:
        return gap
    if gap <= 9:
        return 5
    return 6


def repeat_gap_component(draws):
    sample = draws[-2000:]
    probabilities = np.empty((3, 10), dtype=float)
    for position in range(3):
        exposures = np.zeros((10, 7), dtype=float)
        events = np.zeros((10, 7), dtype=float)
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
        hazards = np.empty(10, dtype=float)
        next_index = len(sample)
        for digit in range(10):
            gap = next_index - last_seen[digit] - 1 if last_seen[digit] >= 0 else 10
            bucket = gap_bucket(int(gap))
            hazards[digit] = (events[digit, bucket] + 2.0) / (exposures[digit, bucket] + 20.0)
        probabilities[position] = hazards / hazards.sum()
    return normalize(np.prod(probabilities[np.arange(3)[:, None], DIGITS.T], axis=0))


def change_point_component(draws):
    recent = draws[-50:]
    reference = draws[-250:-50] if len(draws) >= 250 else draws[:-50]
    if len(reference) < 20:
        reference = draws
    probs = np.empty((3, 10), dtype=float)
    for position in range(3):
        recent_counts = np.bincount(recent[:, position], minlength=10).astype(float) + 10.0
        reference_counts = np.bincount(reference[:, position], minlength=10).astype(float) + 20.0
        recent_rate = recent_counts / recent_counts.sum()
        reference_rate = reference_counts / reference_counts.sum()
        bounded_shift = np.clip(np.log(recent_rate / reference_rate), -0.35, 0.35)
        adjusted = reference_rate * np.exp(bounded_shift)
        probs[position] = adjusted / adjusted.sum()
    return normalize(np.prod(probs[np.arange(3)[:, None], DIGITS.T], axis=0))


def probability_model(numbers):
    draws = np.asarray(numbers, dtype=int)
    if draws.ndim != 2 or draws.shape[1] != 3 or len(draws) < 250:
        raise ValueError("CHALLENGER-V2 requires at least 250 preceding same-stream draws")
    if np.any(draws < 0) or np.any(draws > 9):
        raise ValueError("History contains invalid digits")
    components = {
        "position": position_component(draws),
        "pairs": pair_component(draws),
        "transition": transition_component(draws),
        "repeat_gap": repeat_gap_component(draws),
        "change_point": change_point_component(draws),
    }
    signal = normalize(sum(COMPONENT_WEIGHTS[name] * components[name] for name in COMPONENT_WEIGHTS))
    final = (1.0 - SIGNAL_WEIGHT) / 1000.0 + SIGNAL_WEIGHT * signal
    final = normalize(final)
    return final, components


def top_numbers(probabilities, count=3):
    tie_break = np.random.default_rng(SEED).random(1000)
    order = np.lexsort((tie_break, -np.asarray(probabilities)))
    return [f"{number:03d}" for number in order[:count]]


def select_boxes(probabilities):
    p = np.asarray(probabilities)
    ranked = sorted(BOX_GROUPS, key=lambda key: (-float(p[BOX_GROUPS[key]].sum()), key))[:3]
    return [{
        "digits": key,
        "ways": len(BOX_GROUPS[key]),
        "probability": float(p[BOX_GROUPS[key]].sum()),
        "gross_prize_usd": 160 if len(BOX_GROUPS[key]) == 3 else 80,
    } for key in ranked]


def controls(date, draw_type, boxes):
    seed = int(hashlib.sha256(f"{VERSION}|{date}|{draw_type}|{SEED}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    straight = [f"{number:03d}" for number in rng.choice(1000, 3, replace=False)]
    random_boxes = []
    used = set()
    for box in boxes:
        eligible = [key for key, values in BOX_GROUPS.items() if len(values) == box["ways"] and key not in used]
        key = str(rng.choice(eligible))
        used.add(key)
        random_boxes.append({
            "digits": key,
            "ways": box["ways"],
            "gross_prize_usd": box["gross_prize_usd"],
        })
    return straight, random_boxes


def model_history(forecast):
    history = pd.read_csv(PAPER / "frozen/history.csv", dtype={"number": str})
    history = history[(history.draw_type == forecast["draw_type"]) & (history.date <= forecast["training_through"])][["date", "number"]]
    additions = []
    for result_path in (PAPER / "results").glob(f"*_{forecast['draw_type']}.json"):
        result = json.loads(result_path.read_text())
        if result["date"] <= forecast["training_through"] and result["date"] not in set(history.date):
            additions.append({"date": result["date"], "number": result["number"]})
    if additions:
        history = pd.concat([history, pd.DataFrame(additions)], ignore_index=True)
    history = history.sort_values("date").drop_duplicates("date", keep="last")
    if history.date.max() != forecast["training_through"]:
        raise ValueError("Verified history does not reach the forecast training cutoff")
    digits = np.array([[int(value) for value in number.zfill(3)] for number in history.number], dtype=int)
    snapshot = json.dumps(history.to_dict("records"), sort_keys=True, separators=(",", ":")).encode()
    return digits, hashlib.sha256(snapshot).hexdigest()


def box_payout(boxes, actual):
    key = "".join(sorted(actual))
    return sum(box["gross_prize_usd"] for box in boxes if box["digits"] == key)


def append_audit(path):
    journal_path = OUT / "audit.jsonl"
    previous = "0" * 64
    known = set()
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            row = json.loads(line)
            chain_hash = row.pop("chain_hash")
            if row["previous"] != previous or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() != chain_hash:
                raise ValueError("CHALLENGER-V2 audit chain mismatch")
            target = OUT / row["file"]
            if digest(target) != row["sha256"]:
                raise ValueError("CHALLENGER-V2 registered record changed: " + row["file"])
            previous = chain_hash
            known.add(row["file"])
    relative = path.relative_to(OUT).as_posix()
    if relative in known:
        return
    row = {"file": relative, "sha256": digest(path), "previous": previous, "logged_utc": utcnow().isoformat()}
    chain_hash = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    row["chain_hash"] = chain_hash
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def initialize():
    for name in ("forecasts", "results"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["code_sha256"] != digest(Path(__file__)):
            raise ValueError("CHALLENGER-V2 code changed; create a new prospective version")
        return protocol
    protocol = {
        "version": VERSION,
        "created_utc": utcnow().isoformat(),
        "status": "LOCKED_PROSPECTIVE",
        "code_sha256": digest(Path(__file__)),
        "seed": SEED,
        "signal_weight": SIGNAL_WEIGHT,
        "uniform_weight": 1.0 - SIGNAL_WEIGHT,
        "component_weights": COMPONENT_WEIGHTS,
        "features": {
            "position": "Smoothed position frequencies over the previous 50, 200 and all same-stream draws.",
            "pairs": "Smoothed H-T, T-U and H-U digit-pair frequencies over at most 500 preceding same-stream draws.",
            "transition": "Smoothed position-specific next-digit transitions conditioned on the immediately previous draw.",
            "repeat_gap": "Smoothed empirical repeat hazard by position and gap buckets 0,1,2,3,4,5-9,10+.",
            "change_point": "Recent 50 versus preceding 200 position frequencies, with log shifts capped at +/-0.35.",
        },
        "selection": "Three highest-probability straight numbers and three highest-probability unordered 3-way/6-way Box classes. Triples excluded from Box selections.",
        "controls": "Three seeded random distinct straights and three seeded random boxes with identical 3-way/6-way composition, coverage and cost.",
        "registration": "Register only before 12:45 Day / 18:45 Night America/New_York. Use only verified preceding draws from the same stream. Exclusive-create records; never backfill, overwrite or retune this version.",
        "costs": {"straight_cost_per_draw_usd": 3, "box_cost_per_draw_usd": 3, "straight_gross_prize_usd": 500, "box_3way_gross_usd": 160, "box_6way_gross_usd": 80},
        "evaluation": "Report straight hits, Box hits, model and uniform log loss, baseline log loss, paired matched-control results, gross prizes and net. First descriptive checkpoint after 100 scored Challenger Box drawings. Final comparison after 200 scored draws per stream. No promotion based on an interim balance or isolated hit.",
        "limitations": "A lottery drawing may be independent of prior results. This challenger is an experiment, not evidence of an advantage and not a recommendation to buy tickets.",
    }
    write_once(protocol_path, protocol)
    return protocol


def register_forecasts(protocol):
    now = utcnow()
    for baseline_path in sorted((PAPER / "forecasts").glob("*.json")):
        baseline = json.loads(baseline_path.read_text())
        target = OUT / "forecasts" / baseline_path.name
        if target.exists() or now >= deadline(baseline["date"], baseline["draw_type"]):
            continue
        numbers, history_sha256 = model_history(baseline)
        probabilities, components = probability_model(numbers)
        boxes = select_boxes(probabilities)
        random_straights, random_boxes = controls(baseline["date"], baseline["draw_type"], boxes)
        registered = utcnow()
        if registered >= deadline(baseline["date"], baseline["draw_type"]):
            continue
        record = {
            "version": VERSION,
            "date": baseline["date"],
            "draw_type": baseline["draw_type"],
            "registered_utc": registered.isoformat(),
            "deadline_local": deadline(baseline["date"], baseline["draw_type"]).isoformat(),
            "training_through": baseline["training_through"],
            "history_sha256": history_sha256,
            "protocol_sha256": digest(OUT / "protocol.json"),
            "baseline_forecast_sha256": digest(baseline_path),
            "probabilities": probabilities.tolist(),
            "component_top3": {name: top_numbers(value) for name, value in components.items()},
            "straight_picks": top_numbers(probabilities),
            "random_straight_picks": random_straights,
            "box_picks": boxes,
            "random_box_picks": random_boxes,
            "straight_cost_usd": 3,
            "box_cost_usd": 3,
        }
        write_once(target, record)
        append_audit(target)


def score_forecasts():
    for forecast_path in sorted((OUT / "forecasts").glob("*.json")):
        forecast = json.loads(forecast_path.read_text())
        baseline_result_path = PAPER / "results" / forecast_path.name
        result_path = OUT / "results" / forecast_path.name
        if not baseline_result_path.exists() or result_path.exists():
            continue
        baseline_result = json.loads(baseline_result_path.read_text())
        baseline_forecast_path = PAPER / "forecasts" / forecast_path.name
        if digest(baseline_forecast_path) != forecast["baseline_forecast_sha256"]:
            raise ValueError("Referenced baseline forecast changed")
        actual = baseline_result["number"]
        probability = float(forecast["probabilities"][int(actual)])
        baseline_forecast = json.loads(baseline_forecast_path.read_text())
        model_box_gross = box_payout(forecast["box_picks"], actual)
        random_box_gross = box_payout(forecast["random_box_picks"], actual)
        result = {
            "date": forecast["date"],
            "draw_type": forecast["draw_type"],
            "actual": actual,
            "scored_utc": utcnow().isoformat(),
            "forecast_sha256": digest(forecast_path),
            "verified_baseline_result_sha256": digest(baseline_result_path),
            "straight_hit": actual in forecast["straight_picks"],
            "random_straight_hit": actual in forecast["random_straight_picks"],
            "box_hit": model_box_gross > 0,
            "random_box_hit": random_box_gross > 0,
            "model_log_loss": float(-np.log(probability)),
            "uniform_log_loss": float(np.log(1000.0)),
            "baseline_log_loss": float(-np.log(baseline_forecast["probabilities"][int(actual)])),
            "straight_gross_usd": 500 if actual in forecast["straight_picks"] else 0,
            "straight_net_usd": (500 if actual in forecast["straight_picks"] else 0) - 3,
            "random_straight_net_usd": (500 if actual in forecast["random_straight_picks"] else 0) - 3,
            "box_gross_usd": model_box_gross,
            "box_net_usd": model_box_gross - 3,
            "random_box_gross_usd": random_box_gross,
            "random_box_net_usd": random_box_gross - 3,
            "best_position_matches": max(sum(a == b for a, b in zip(pick, actual)) for pick in forecast["straight_picks"]),
            "best_digit_overlap": max(sum((Counter(pick) & Counter(actual)).values()) for pick in forecast["straight_picks"]),
        }
        write_once(result_path, result)
        append_audit(result_path)


def render(protocol):
    rows = []
    for forecast_path in sorted((OUT / "forecasts").glob("*.json")):
        forecast = json.loads(forecast_path.read_text())
        result_path = OUT / "results" / forecast_path.name
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        rows.append({
            "date": forecast["date"],
            "draw_type": forecast["draw_type"],
            "registered_utc": forecast["registered_utc"],
            "straight_picks": forecast["straight_picks"],
            "box_picks": forecast["box_picks"],
            "result": result,
        })
    scored = [row["result"] for row in rows if row["result"]]
    totals = {
        "scored_draws": len(scored),
        "straight_hits": sum(result["straight_hit"] for result in scored),
        "random_straight_hits": sum(result["random_straight_hit"] for result in scored),
        "box_hits": sum(result["box_hit"] for result in scored),
        "random_box_hits": sum(result["random_box_hit"] for result in scored),
        "straight_net_usd": sum(result["straight_net_usd"] for result in scored),
        "random_straight_net_usd": sum(result["random_straight_net_usd"] for result in scored),
        "box_net_usd": sum(result["box_net_usd"] for result in scored),
        "random_box_net_usd": sum(result["random_box_net_usd"] for result in scored),
        "mean_model_log_loss": float(np.mean([result["model_log_loss"] for result in scored])) if scored else None,
        "mean_uniform_log_loss": float(np.mean([result["uniform_log_loss"] for result in scored])) if scored else None,
        "mean_baseline_log_loss": float(np.mean([result["baseline_log_loss"] for result in scored])) if scored else None,
    }
    summary = {
        "version": VERSION,
        "status": protocol["status"],
        "updated_utc": utcnow().isoformat(),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "totals": totals,
        "checkpoint": {"first_box_checkpoint_draws": 100, "final_draws_per_stream": 200},
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = "# CHALLENGER-V2 — locked prospective study\n\n"
    text += "This separate challenger is 85% uniform and 15% historical signal. It does not replace the baseline. Paper selections only; no tickets purchased.\n\n"
    text += "| Date | Drawing | Three straights | Three boxes | Actual | Straight | Box |\n|---|---|---|---|---|---|---|\n"
    for row in rows:
        result = row["result"]
        boxes = ", ".join(f"{box['digits']} ({box['ways']}-way)" for box in row["box_picks"])
        text += f"| {row['date']} | {row['draw_type']} | {', '.join(row['straight_picks'])} | {boxes} | {result['actual'] if result else 'Pending'} | {result['straight_hit'] if result else 'Pending'} | {result['box_hit'] if result else 'Pending'} |\n"
    text += "\nFirst descriptive checkpoint: 100 scored Challenger Box drawings. Final comparison: 200 scored draws per stream. Interim balances do not establish an advantage.\n"
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(text)
    return summary


def main():
    protocol = initialize()
    register_forecasts(protocol)
    score_forecasts()
    render(protocol)


if __name__ == "__main__":
    main()
