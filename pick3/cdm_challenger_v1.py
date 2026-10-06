"""Locked prospective CDM-CHALLENGER-V1 for South Carolina Pick 3.

This research-inspired challenger combines an exchangeable
Dirichlet-multinomial forecast with position-specific Dirichlet posterior
predictions. It runs beside the existing studies and never backfills or
rewrites a registered forecast.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from paper_track import PAPER, ROOT, deadline, digest, write_once


OUT = ROOT / "cdm_challenger_v1"
VERSION = "CDM-CHALLENGER-V1"
SEED = 20261006
POSITION_PRIOR_PER_DIGIT = 10.0
GLOBAL_PRIOR_PER_DIGIT = 1.0
CDM_WEIGHT = 0.50
POSITION_WEIGHT = 0.50
MIN_CONCENTRATION = 1.0
MAX_CONCENTRATION = 10000.0
DIGITS = np.array([[n // 100, (n // 10) % 10, n % 10] for n in range(1000)], dtype=int)
BOX_GROUPS: dict[str, list[int]] = {}
for number in range(1000):
    BOX_GROUPS.setdefault("".join(sorted(f"{number:03d}")), []).append(number)
BOX_GROUPS = {key: values for key, values in BOX_GROUPS.items() if len(values) in (3, 6)}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (1000,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid CDM probability vector")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Empty CDM probability vector")
    return values / total


def estimate_cdm_alpha(draws: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate Dirichlet concentration from within-draw digit collisions.

    The Dirichlet-multinomial implies pair collision probability
    C=(a0*sum(p^2)+1)/(a0+1). The method-of-moments solution is fixed in the
    protocol and clipped only for numerical stability.
    """
    counts = np.bincount(draws.ravel(), minlength=10).astype(float) + GLOBAL_PRIOR_PER_DIGIT
    proportions = counts / counts.sum()
    collision = float(np.mean([
        draws[:, 0] == draws[:, 1],
        draws[:, 0] == draws[:, 2],
        draws[:, 1] == draws[:, 2],
    ]))
    independent_collision = float(np.square(proportions).sum())
    denominator = collision - independent_collision
    if denominator <= 1e-12:
        concentration = MAX_CONCENTRATION
    else:
        concentration = (1.0 - collision) / denominator
        concentration = float(np.clip(concentration, MIN_CONCENTRATION, MAX_CONCENTRATION))
    alpha = np.maximum(proportions * concentration, 1e-9)
    diagnostics = {
        "concentration": concentration,
        "observed_pair_collision": collision,
        "independent_pair_collision": independent_collision,
        "digit_proportions": proportions.tolist(),
    }
    return alpha, diagnostics


def cdm_component(draws: np.ndarray) -> tuple[np.ndarray, dict]:
    alpha, diagnostics = estimate_cdm_alpha(draws)
    alpha0 = float(alpha.sum())
    scores = np.empty(1000, dtype=float)
    denominator = alpha0 * (alpha0 + 1.0) * (alpha0 + 2.0)
    for number, digits in enumerate(DIGITS):
        digit_counts = np.bincount(digits, minlength=10)
        numerator = 1.0
        for digit, count in enumerate(digit_counts):
            for step in range(int(count)):
                numerator *= alpha[digit] + step
        scores[number] = numerator / denominator
    return normalize(scores), diagnostics


def position_component(draws: np.ndarray) -> np.ndarray:
    probabilities = np.empty((3, 10), dtype=float)
    for position in range(3):
        counts = np.bincount(draws[:, position], minlength=10).astype(float)
        counts += POSITION_PRIOR_PER_DIGIT
        probabilities[position] = counts / counts.sum()
    return normalize(np.prod(probabilities[np.arange(3)[:, None], DIGITS.T], axis=0))


def probability_model(numbers) -> tuple[np.ndarray, dict, dict]:
    draws = np.asarray(numbers, dtype=int)
    if draws.ndim != 2 or draws.shape[1] != 3 or len(draws) < 250:
        raise ValueError("CDM-CHALLENGER-V1 requires at least 250 preceding same-stream draws")
    if np.any(draws < 0) or np.any(draws > 9):
        raise ValueError("History contains invalid digits")
    cdm, diagnostics = cdm_component(draws)
    position = position_component(draws)
    # A fixed logarithmic pool preserves positive support and gives both
    # predeclared components equal influence without fitting a blend to results.
    combined = normalize(np.exp(CDM_WEIGHT * np.log(cdm) + POSITION_WEIGHT * np.log(position)))
    return combined, {"cdm": cdm, "position": position}, diagnostics


def top_numbers(probabilities, count: int = 3) -> list[str]:
    tie_break = np.random.default_rng(SEED).random(1000)
    order = np.lexsort((tie_break, -np.asarray(probabilities)))
    return [f"{number:03d}" for number in order[:count]]


def select_boxes(probabilities) -> list[dict]:
    p = np.asarray(probabilities)
    ranked = sorted(BOX_GROUPS, key=lambda key: (-float(p[BOX_GROUPS[key]].sum()), key))[:3]
    return [{
        "digits": key,
        "ways": len(BOX_GROUPS[key]),
        "probability": float(p[BOX_GROUPS[key]].sum()),
        "gross_prize_usd": 160 if len(BOX_GROUPS[key]) == 3 else 80,
    } for key in ranked]


def controls(date: str, draw_type: str, boxes: list[dict]) -> tuple[list[str], list[dict]]:
    seed = int(hashlib.sha256(f"{VERSION}|{date}|{draw_type}|{SEED}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    straights = [f"{number:03d}" for number in rng.choice(1000, 3, replace=False)]
    random_boxes = []
    used = set()
    for box in boxes:
        eligible = [key for key, values in BOX_GROUPS.items() if len(values) == box["ways"] and key not in used]
        key = str(rng.choice(eligible))
        used.add(key)
        random_boxes.append({"digits": key, "ways": box["ways"], "gross_prize_usd": box["gross_prize_usd"]})
    return straights, random_boxes


def model_history(forecast):
    history = pd.read_csv(PAPER / "frozen/history.csv", dtype={"number": str})
    history = history[(history.draw_type == forecast["draw_type"]) & (history.date <= forecast["training_through"])][["date", "number"]]
    additions = []
    for result_path in (PAPER / "results").glob(f"*_{forecast['draw_type']}.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["date"] <= forecast["training_through"]:
            additions.append({"date": result["date"], "number": result["number"]})
    if additions:
        history = pd.concat([history, pd.DataFrame(additions)], ignore_index=True)
    history = history.sort_values("date").drop_duplicates("date", keep="last")
    if history.date.max() != forecast["training_through"]:
        raise ValueError("Verified history does not reach the forecast training cutoff")
    digits = np.array([[int(value) for value in number.zfill(3)] for number in history.number], dtype=int)
    snapshot = json.dumps(history.to_dict("records"), sort_keys=True, separators=(",", ":")).encode()
    return digits, hashlib.sha256(snapshot).hexdigest()


def box_payout(boxes: list[dict], actual: str) -> int:
    key = "".join(sorted(actual))
    return sum(box["gross_prize_usd"] for box in boxes if box["digits"] == key)


def append_audit(path: Path) -> None:
    journal_path = OUT / "audit.jsonl"
    previous = "0" * 64
    known = set()
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            chain_hash = row.pop("chain_hash")
            if row["previous"] != previous or hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() != chain_hash:
                raise ValueError("CDM-CHALLENGER-V1 audit chain mismatch")
            target = OUT / row["file"]
            if digest(target) != row["sha256"]:
                raise ValueError("CDM-CHALLENGER-V1 registered record changed: " + row["file"])
            previous = chain_hash
            known.add(row["file"])
    relative = path.relative_to(OUT).as_posix()
    if relative in known:
        return
    row = {"file": relative, "sha256": digest(path), "previous": previous, "logged_utc": utcnow().isoformat()}
    row["chain_hash"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def initialize() -> dict:
    for name in ("forecasts", "results"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol["code_sha256"] != digest(Path(__file__)):
            raise ValueError("CDM-CHALLENGER-V1 code changed; create a new prospective version")
        return protocol
    protocol = {
        "version": VERSION,
        "created_utc": utcnow().isoformat(),
        "status": "LOCKED_PROSPECTIVE",
        "code_sha256": digest(Path(__file__)),
        "research_basis": "Research-inspired implementation informed by Nkomozake, Predicting Winning Lottery Numbers, arXiv:2403.12836. This is an independent, fully specified implementation; it does not reproduce or validate that paper's claimed strategy.",
        "seed": SEED,
        "model": {
            "cdm_weight": CDM_WEIGHT,
            "position_weight": POSITION_WEIGHT,
            "position_prior_per_digit": POSITION_PRIOR_PER_DIGIT,
            "global_prior_per_digit": GLOBAL_PRIOR_PER_DIGIT,
            "concentration_estimator": "Fixed method-of-moments from within-draw pair collision frequency, clipped to [1, 10000].",
            "pool": "Equal-weight logarithmic pool of exchangeable Dirichlet-multinomial sequence probabilities and position-specific Dirichlet-categorical posterior probabilities.",
        },
        "selection": "Three highest-probability straights and three highest-probability unordered 3-way/6-way Box classes; triples excluded from Box selections.",
        "controls": "Three seeded random distinct straights and seeded random boxes with identical 3-way/6-way composition, coverage and cost.",
        "registration": "Register only before 12:45 Day / 18:45 Night America/New_York using verified preceding same-stream draws. Exclusive-create records; never backfill, overwrite or retune this version.",
        "costs": {"straight_cost_per_draw_usd": 3, "box_cost_per_draw_usd": 3, "straight_gross_prize_usd": 500, "box_3way_gross_usd": 160, "box_6way_gross_usd": 80},
        "evaluation": "Compare exact and Box hits, log loss, gross prizes and net against uniform, original baseline, CHALLENGER-V2 and coverage-matched seeded random controls. Descriptive checkpoint at 100 scored drawings; final comparison at 200 scored draws per stream.",
        "limitations": "A fair lottery may be independent of history. This paper-only experiment is not evidence of an advantage or a recommendation to buy tickets. Progressive staking is prohibited.",
    }
    write_once(protocol_path, protocol)
    return protocol


def register_forecasts(protocol: dict) -> None:
    now = utcnow()
    for baseline_path in sorted((PAPER / "forecasts").glob("*.json")):
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        target = OUT / "forecasts" / baseline_path.name
        if target.exists() or now >= deadline(baseline["date"], baseline["draw_type"]):
            continue
        numbers, history_sha256 = model_history(baseline)
        probabilities, components, diagnostics = probability_model(numbers)
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
            "component_top3": {name: top_numbers(values) for name, values in components.items()},
            "cdm_diagnostics": diagnostics,
            "straight_picks": top_numbers(probabilities),
            "random_straight_picks": random_straights,
            "box_picks": boxes,
            "random_box_picks": random_boxes,
            "straight_cost_usd": 3,
            "box_cost_usd": 3,
        }
        write_once(target, record)
        append_audit(target)


def score_forecasts() -> None:
    for forecast_path in sorted((OUT / "forecasts").glob("*.json")):
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        baseline_result_path = PAPER / "results" / forecast_path.name
        result_path = OUT / "results" / forecast_path.name
        if not baseline_result_path.exists() or result_path.exists():
            continue
        if digest(PAPER / "forecasts" / forecast_path.name) != forecast["baseline_forecast_sha256"]:
            raise ValueError("Referenced baseline forecast changed")
        baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
        baseline_forecast = json.loads((PAPER / "forecasts" / forecast_path.name).read_text(encoding="utf-8"))
        actual = baseline_result["number"]
        model_box_gross = box_payout(forecast["box_picks"], actual)
        random_box_gross = box_payout(forecast["random_box_picks"], actual)
        result = {
            "date": forecast["date"], "draw_type": forecast["draw_type"], "actual": actual,
            "scored_utc": utcnow().isoformat(), "forecast_sha256": digest(forecast_path),
            "verified_baseline_result_sha256": digest(baseline_result_path),
            "straight_hit": actual in forecast["straight_picks"],
            "random_straight_hit": actual in forecast["random_straight_picks"],
            "box_hit": model_box_gross > 0, "random_box_hit": random_box_gross > 0,
            "model_log_loss": float(-np.log(forecast["probabilities"][int(actual)])),
            "uniform_log_loss": float(np.log(1000.0)),
            "baseline_log_loss": float(-np.log(baseline_forecast["probabilities"][int(actual)])),
            "straight_gross_usd": 500 if actual in forecast["straight_picks"] else 0,
            "straight_net_usd": (500 if actual in forecast["straight_picks"] else 0) - 3,
            "random_straight_net_usd": (500 if actual in forecast["random_straight_picks"] else 0) - 3,
            "box_gross_usd": model_box_gross, "box_net_usd": model_box_gross - 3,
            "random_box_gross_usd": random_box_gross, "random_box_net_usd": random_box_gross - 3,
            "best_position_matches": max(sum(a == b for a, b in zip(pick, actual)) for pick in forecast["straight_picks"]),
            "best_digit_overlap": max(sum((Counter(pick) & Counter(actual)).values()) for pick in forecast["straight_picks"]),
        }
        write_once(result_path, result)
        append_audit(result_path)


def render(protocol: dict) -> dict:
    rows = []
    for path in sorted((OUT / "forecasts").glob("*.json")):
        forecast = json.loads(path.read_text(encoding="utf-8"))
        result_path = OUT / "results" / path.name
        rows.append({
            "date": forecast["date"], "draw_type": forecast["draw_type"],
            "registered_utc": forecast["registered_utc"], "straight_picks": forecast["straight_picks"],
            "box_picks": forecast["box_picks"],
            "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None,
        })
    scored = [row["result"] for row in rows if row["result"]]
    totals = {
        "scored_draws": len(scored),
        "straight_hits": sum(row["straight_hit"] for row in scored),
        "random_straight_hits": sum(row["random_straight_hit"] for row in scored),
        "box_hits": sum(row["box_hit"] for row in scored),
        "random_box_hits": sum(row["random_box_hit"] for row in scored),
        "straight_net_usd": sum(row["straight_net_usd"] for row in scored),
        "random_straight_net_usd": sum(row["random_straight_net_usd"] for row in scored),
        "box_net_usd": sum(row["box_net_usd"] for row in scored),
        "random_box_net_usd": sum(row["random_box_net_usd"] for row in scored),
        "mean_model_log_loss": float(np.mean([row["model_log_loss"] for row in scored])) if scored else None,
        "mean_uniform_log_loss": float(np.mean([row["uniform_log_loss"] for row in scored])) if scored else None,
        "mean_baseline_log_loss": float(np.mean([row["baseline_log_loss"] for row in scored])) if scored else None,
    }
    summary = {"version": VERSION, "status": protocol["status"], "updated_utc": utcnow().isoformat(),
               "protocol_sha256": digest(OUT / "protocol.json"), "totals": totals,
               "checkpoint": {"descriptive_draws": 100, "final_draws_per_stream": 200}, "rows": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = "# CDM-CHALLENGER-V1 — locked prospective study\n\n"
    text += "A research-inspired hierarchical Dirichlet-multinomial model running beside the existing systems. Paper selections only; no progressive staking or ticket purchases.\n\n"
    text += "| Date | Drawing | Three straights | Three boxes | Actual | Straight | Box |\n|---|---|---|---|---|---|---|\n"
    for row in rows:
        result = row["result"]
        boxes = ", ".join(f"{box['digits']} ({box['ways']}-way)" for box in row["box_picks"])
        text += f"| {row['date']} | {row['draw_type']} | {', '.join(row['straight_picks'])} | {boxes} | {result['actual'] if result else 'Pending'} | {result['straight_hit'] if result else 'Pending'} | {result['box_hit'] if result else 'Pending'} |\n"
    text += "\nDescriptive checkpoint: 100 scored drawings. Final comparison: 200 scored draws per stream. Interim outcomes do not establish an advantage.\n"
    (OUT / "README.md").write_text(text, encoding="utf-8")
    print(text)
    return summary


def main() -> None:
    protocol = initialize()
    register_forecasts(protocol)
    score_forecasts()
    render(protocol)


if __name__ == "__main__":
    main()
