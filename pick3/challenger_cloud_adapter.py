"""Verified-history adapter for the byte-locked CHALLENGER-V2 model."""
import hashlib
import json

import numpy as np
import pandas as pd

import challenger_v2 as locked


def model_history(forecast):
    history = pd.read_csv(locked.PAPER / "frozen/history.csv", dtype={"number": str})
    history = history[(history.draw_type == forecast["draw_type"]) & (history.date <= forecast["training_through"])][["date", "number"]]
    additions = []
    ledger_path = locked.PAPER / "verified_history.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        additions.extend(
            {"date": row["date"], "number": row["number"]}
            for row in ledger.get("rows", [])
            if row["draw_type"] == forecast["draw_type"] and row["date"] <= forecast["training_through"]
        )
    for result_path in (locked.PAPER / "results").glob(f"*_{forecast['draw_type']}.json"):
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


locked.model_history = model_history
locked.main()
