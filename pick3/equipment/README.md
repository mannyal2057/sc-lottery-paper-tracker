# Machine and ball-set context

Support added September 15, 2026. Status: awaiting sourced per-drawing IDs; no equipment model is active and no predictions have changed.

Public research found SCEL's 2020 drawing-machine procurement document, which describes randomly selected machines/ball sets, pre-tests, and ball weighing. This is evidence about procedures at that time, not a current draw-by-draw assignment feed. Searches of SCEL's Pick 3 page, player-protection FAQ and procurement materials did not locate an ID log. Do not infer IDs from the winning number, machine appearance, day of week or weather.

Sources checked:
- https://www.sceducationlottery.com/Documents/lottery/Procurement/04212020DRAWMACHRFP.pdf
- https://www.sceducationlottery.com/FAQ/PlayerProtection
- https://www.sceducationlottery.com/Games/Pick3

Run `python equipment_context.py` after the other daily scripts. It writes equipment/summary.json with nulls for missing IDs, coverage of registered paper draws, and descriptive digit counts where sourced mappings exist. Baseline and WEATHER-V1 files remain untouched.

## Import format

One JSON record per date, drawing and position in equipment/records. Positions H, T and U mean hundreds, tens and units. Repeat a machine ID across positions only when the source establishes it; record distinct ball-set/chamber IDs if documented. Preserve IDs as strings including leading zeros.

Required fields: date (YYYY-MM-DD), draw_type (Day or Night), position (H/T/U), machine_id, ball_set_id, source_url (public HTTPS), source_file (relative path inside equipment), source_sha256, public_available_utc, retrieved_utc (timezone-aware ISO timestamps), mapping_verified (true only after checking the source explicitly maps these IDs to that drawing and position).

Do not fill a record with placeholders and mark it verified. Download the actual public source into equipment/sources and hash its exact bytes. A hash verifies file consistency, not the truth of its contents; review the source mapping manually. Duplicate mappings and invalid sources are rejected. Records captured after the cutoff may inform descriptive research but cannot support a prospective claim for that drawing. All three positions must have captured pre-cutoff mappings before a future model could use the draw's IDs. This does not establish sufficient training sample size or model effectiveness.

Once data exist, evaluate a new named equipment model chronologically against matched controls. Use shrinkage for sparse equipment groups and account for multiple comparisons. Never silently modify original forecasts. No historical ID assignments or future equipment selections have been invented.

The records-request draft is saved locally and has NOT been sent. It is the next route to ask whether a public log exists, whether records can be released, and when IDs become available. Obtaining historical IDs alone would not reveal the machine and ball set selected for the next drawing.
