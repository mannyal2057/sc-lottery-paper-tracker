"""Build the public, static SC Pick 3 and Pick 4 tracking dashboard."""
import html
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).parent
SITE = ROOT / "site"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def cloud_notice():
    path = ROOT / "cloud_status.json"
    if not path.exists():
        return ""
    status = json.loads(path.read_text(encoding="utf-8"))
    unavailable = []
    if not status.get("pick3_source", {}).get("ok", True):
        unavailable.append("Pick 3")
    if not status.get("pick4_source", {}).get("ok", True):
        unavailable.append("Pick 4")
    failed_supporting = [
        name.removesuffix(".py")
        for name, state in status.get("pick3_supporting", {}).items()
        if not state.get("ok", True)
    ]
    if not unavailable and not failed_supporting:
        return ""
    parts = []
    if unavailable:
        parts.append(f"The {' and '.join(unavailable)} public result source was temporarily unavailable")
    if failed_supporting:
        parts.append("these study updates need attention: " + ", ".join(failed_supporting))
    message = ". ".join(parts) + ". Prior verified records remain intact; unresolved outcomes stay pending."
    return f'<div class="notice warning"><strong>Source status:</strong> {html.escape(message)}</div>'


def newest_pending(rows):
    pending = [row for row in rows if not row.get("result")]
    if not pending:
        return []
    latest = max(row["date"] for row in pending)
    return [row for row in pending if row["date"] == latest]


def cards(rows, game):
    blocks = []
    for row in rows:
        straight_key = "straight_picks"
        picks = row.get(straight_key, [])
        boxes = row.get("box_picks", [])
        box_text = " · ".join(f"{box['digits']} <small>{box['ways']}-way</small>" for box in boxes)
        blocks.append(f"""
        <article class="draw-card">
          <div class="eyebrow">{html.escape(row['date'])} · {html.escape(row['draw_type'])}</div>
          <h3>{game} registered selections</h3>
          <div class="label">Straight</div><div class="numbers">{' · '.join(map(html.escape, picks))}</div>
          <div class="label">Box</div><div class="numbers boxes">{box_text}</div>
          <div class="pending">Pending verified result</div>
        </article>""")
    return "".join(blocks) or '<div class="empty">No pending registered selections.</div>'


def metric(label, value, note=""):
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong><small>{html.escape(note)}</small></div>'


def recent_table(rows, digits):
    completed = [row for row in rows if row.get("result")][-12:][::-1]
    body = []
    for row in completed:
        result = row["result"]
        body.append(
            "<tr>"
            f"<td>{html.escape(row['date'])}</td><td>{html.escape(row['draw_type'])}</td>"
            f"<td class='mono'>{html.escape(', '.join(row['straight_picks']))}</td>"
            f"<td class='mono actual'>{html.escape(result['actual'])}</td>"
            f"<td>{'Yes' if result['straight_hit'] else 'No'}</td><td>{'Yes' if result['box_hit'] else 'No'}</td>"
            "</tr>"
        )
    if not body:
        return '<div class="empty">The prospective study has started; verified results will appear here.</div>'
    return f"""<div class="table-wrap"><table><thead><tr><th>Date</th><th>Draw</th><th>Three picks</th><th>Actual</th><th>Straight</th><th>Box</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"""


def pick3_result_status(source_status):
    awaiting = source_status.get("awaiting_archive_confirmation", [])
    if not awaiting:
        return ""
    blocks = []
    for row in awaiting:
        blocks.append(
            '<article class="result-card pending-result">'
            f'<div><span class="eyebrow">{html.escape(row["date"])} · {html.escape(row["draw_type"])}</span>'
            f'<strong class="reported-number">{html.escape(row["official_number"])}</strong></div>'
            '<div><strong>Reported by the official source</strong><small>Awaiting independent archive confirmation · not scored yet</small></div>'
            '</article>'
        )
    return '<h2>Latest Pick 3 result status</h2><div class="result-list">' + "".join(blocks) + "</div>"


def pick3_history(rows):
    completed = [row for row in rows if row.get("result")][-14:][::-1]
    body = []
    for row in completed:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['date'])}</td><td>{html.escape(row['draw_type'])}</td>"
            f"<td class='mono'>{html.escape(', '.join(row['picks']))}</td>"
            f"<td class='mono actual'>{html.escape(row['result'])}</td>"
            f"<td>{'Yes' if row['straight_hit'] else 'No'}</td>"
            f"<td>{'Yes' if row['box_hit'] else 'No'}</td>"
            f"<td>{row['best_position_matches']}/3</td>"
            "</tr>"
        )
    return f"""<div class="table-wrap"><table><thead><tr><th>Date</th><th>Draw</th><th>Original three picks</th><th>Actual</th><th>Straight</th><th>Any order</th><th>Best positions</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"""


def main():
    pick3 = load("pick3/challenger_v2/summary.json")
    cdm = load("pick3/cdm_challenger_v1/summary.json")
    pick3_review = load("pick3/paper/three_pick_review.json")
    pick3_source = load("pick3/paper/source_status.json")
    pick4 = load("pick4/study/summary.json")
    p3_box = load("pick3/box/summary.json")
    p3_scored = [row for row in p3_box["rows"] if row.get("result")]
    p3_box_net = sum(row["result"]["net_usd"] for row in p3_scored)
    p3_random_net = sum(row["result"]["random_net_usd"] for row in p3_scored)
    p3_totals = pick3["totals"]
    cdm_totals = cdm["totals"]
    p4_totals = pick4["totals"]
    built = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    favicon = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%230d2538'/%3E%3Ccircle cx='32' cy='32' r='20' fill='%23f2b84b'/%3E%3Ctext x='32' y='40' text-anchor='middle' font-size='24' font-family='Arial' font-weight='700' fill='%230d2538'%3E34%3C/text%3E%3C/svg%3E"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC Pick 3 & Pick 4 Paper Tracker</title><meta name="description" content="Public prospective paper tracking for South Carolina Pick 3 and Pick 4.">
<link rel="icon" type="image/svg+xml" href="{favicon}">
<style>
:root{{--ink:#102638;--muted:#617283;--paper:#f5f8fa;--card:#fff;--line:#d9e1e7;--navy:#0d2538;--gold:#f2b84b;--blue:#2c6e9f;--green:#19704a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{background:var(--navy);color:white;padding:1rem 0;border-bottom:5px solid var(--gold)}}.bar,main,footer{{width:min(1180px,calc(100% - 2rem));margin:auto}}.bar{{display:flex;align-items:center;justify-content:space-between;gap:1rem}}.brand{{font-weight:800;font-size:1.15rem;letter-spacing:.01em}}.stamp{{font-size:.83rem;color:#c7d3dc}}
main{{padding:1.5rem 0 3rem}}.notice{{background:#eaf3f8;border:1px solid #c4dce9;border-left:5px solid var(--blue);padding:.8rem 1rem;margin-bottom:1.5rem;border-radius:8px}}.notice.warning{{background:#fff4dd;border-color:#efd49b;border-left-color:#b26b00}}h1{{font-size:clamp(1.8rem,4vw,3rem);line-height:1.05;margin:.2rem 0 .5rem}}h2{{font-size:1.35rem;margin:2rem 0 .75rem}}.intro{{color:var(--muted);max-width:770px;margin:0 0 1.3rem}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}}.draw-card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.15rem;box-shadow:0 8px 24px rgba(23,48,68,.06)}}.eyebrow,.label{{text-transform:uppercase;letter-spacing:.09em;font-size:.75rem;font-weight:750;color:var(--blue)}}h3{{font-size:1.05rem;margin:.2rem 0 1rem}}.label{{margin-top:.8rem;color:var(--muted)}}.numbers{{font:800 clamp(1.25rem,3vw,1.8rem)/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.03em}}.numbers small{{font:600 .7rem/1 system-ui;color:var(--muted)}}.pending{{display:inline-block;margin-top:1rem;padding:.28rem .55rem;background:#fff5d9;color:#72510d;border-radius:999px;font-size:.78rem;font-weight:700}}.result-list{{display:grid;gap:.7rem}}.result-card{{display:flex;align-items:center;justify-content:space-between;gap:1.2rem;background:white;border:1px solid var(--line);border-radius:12px;padding:1rem 1.15rem}}.result-card.pending-result{{border-left:5px solid var(--gold)}}.reported-number{{display:block;font:800 2rem/1.15 ui-monospace,SFMono-Regular,Consolas,monospace}}.result-card small{{display:block;color:var(--muted);margin-top:.15rem}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem}}.metric{{background:white;border:1px solid var(--line);border-radius:10px;padding:.85rem}}.metric span,.metric small{{display:block;color:var(--muted);font-size:.78rem}}.metric strong{{display:block;font-size:1.45rem;margin:.12rem 0}}.table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;min-width:680px}}th,td{{padding:.72rem .8rem;border-bottom:1px solid var(--line);text-align:left;font-size:.88rem}}th{{background:#edf3f7;color:#435767}}.mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.actual{{font-weight:800}}.empty{{background:white;border:1px dashed var(--line);border-radius:10px;padding:1rem;color:var(--muted)}}footer{{padding:1.2rem 0 2.5rem;color:var(--muted);font-size:.82rem}}
@media(max-width:760px){{.grid,.metrics{{grid-template-columns:1fr}}.bar,.result-card{{align-items:flex-start;flex-direction:column}}main{{padding-top:1rem}}}}
</style></head><body>
<header><div class="bar"><div class="brand">SC Number Lab · Paper Tracker</div><div class="stamp">Updated {built}</div></div></header>
<main><h1>Pick 3 & Pick 4 registered forecasts</h1><p class="intro">Predictions are recorded before each drawing and scored only after public sources agree. These are experiments, not winning guarantees or ticket recommendations.</p>
<div class="notice"><strong>Laptop-free updates:</strong> the public cloud workflow checks four times daily, beginning before each drawing cutoff, and refreshes this page automatically.</div>
{cloud_notice()}
{pick3_result_status(pick3_source)}
<h2>Pick 3 · CHALLENGER-V2</h2><div class="grid">{cards(newest_pending(pick3['rows']), 'Pick 3')}</div>
<h2>Pick 3 scoreboard</h2><div class="metrics">{metric('Challenger scored',p3_totals['scored_draws'],'of first 100 checkpoint')}{metric('Challenger straight hits',p3_totals['straight_hits'])}{metric('Original Box net',f"${p3_box_net:+.0f}",f"Random control ${p3_random_net:+.0f}")}{metric('Original Box draws',len(p3_scored),'coverage matched')}</div>
<h2>Pick 3 · CDM-CHALLENGER-V1</h2><div class="grid">{cards(newest_pending(cdm['rows']), 'CDM Pick 3')}</div>
<h2>CDM scoreboard</h2><div class="metrics">{metric('Scored draws',cdm_totals['scored_draws'],'of 100 descriptive checkpoint')}{metric('Straight hits',cdm_totals['straight_hits'],f"Random {cdm_totals['random_straight_hits']}")}{metric('Box hits',cdm_totals['box_hits'],f"Random {cdm_totals['random_box_hits']}")}{metric('CDM Box net',f"${cdm_totals['box_net_usd']:+.0f}",f"Random ${cdm_totals['random_box_net_usd']:+.0f}")}</div>
<h2>Recent verified Pick 3 results</h2>{pick3_history(pick3_review['rows'])}
<h2>Pick 4 · locked prospective study</h2><div class="grid">{cards(newest_pending(pick4['rows']), 'Pick 4')}</div>
<h2>Pick 4 scoreboard</h2><div class="metrics">{metric('Scored draws',p4_totals['scored_draws'],'of first 100 checkpoint')}{metric('Straight hits',p4_totals['straight_hits'])}{metric('Box hits',p4_totals['box_hits'])}{metric('Random Box hits',p4_totals['random_box_hits'],'same coverage')}</div>
<h2>Recent Pick 4 results</h2>{recent_table(pick4['rows'],4)}
</main><footer>Public paper study · South Carolina Pick 3 and Pick 4 · FIREBALL excluded · No purchases are made.</footer></body></html>"""
    SITE.mkdir(exist_ok=True)
    # Keep the generated artifact byte-stable across Windows development and
    # Linux GitHub Actions runners.
    (SITE / "index.html").write_bytes(page.encode("utf-8"))


if __name__ == "__main__":
    main()
