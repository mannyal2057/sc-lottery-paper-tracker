"""User-facing three-pick slate and next-day review, preserving the ten-pick study."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
from html import escape
import json
from paper_track import PAPER,deadline,write_once,TZ

def main():
    config_path=PAPER/'three_pick_protocol.json'
    if not config_path.exists():
        write_once(config_path,{'created_utc':datetime.now(timezone.utc).isoformat(),
            'rule':'First three ordered model_top10 and first three random_top10 in each pre-draw registered forecast. Exact straight, any-order box and matching digits are reported. Do not change historical selections. Review both draws the following morning; prospective adjustments need a new version.',
            'version':'TOP3-V1'})
    config=json.loads(config_path.read_text());created=datetime.fromisoformat(config['created_utc'])
    rows=[]
    for p in sorted((PAPER/'forecasts').glob('*.json')):
        f=json.loads(p.read_text())
        if created>=deadline(f['date'],f['draw_type']):continue
        picks=f['model_top10'][:3];random=f['random_top10'][:3]
        path=PAPER/'results'/p.name;result=json.loads(path.read_text()) if path.exists() else None
        actual=result['number'] if result else None
        row={'date':f['date'],'draw_type':f['draw_type'],'picks':picks,'random_picks':random,
             'original_registered_utc':f['registered_utc'],'selection_rule_registered_utc':config['created_utc'],
             'result':actual,'straight_hit':actual in picks if actual else None,
             'random_straight_hit':actual in random if actual else None,
             'box_hit':any(sorted(x)==sorted(actual) for x in picks) if actual else None,
             'best_position_matches':max(sum(a==b for a,b in zip(x,actual)) for x in picks) if actual else None,
             'best_digit_overlap':max(sum((Counter(x)&Counter(actual)).values()) for x in picks) if actual else None,
             'review_on':(datetime.fromisoformat(f['date'])+timedelta(days=1)).date().isoformat()}
        rows.append(row)
    summary={'updated_utc':datetime.now(timezone.utc).isoformat(),'version':'TOP3-V1','rows':rows,
             'note':'Three distinct straights have fair hit probability 3/1000 per draw. No ticket purchases. One miss is not evidence that a new model will be better.'}
    (PAPER/'three_pick_review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text='THREE-PICK SLATES — Eastern time\n\n'
    for r in rows:
        text+=r['date']+' '+r['draw_type']+': '+', '.join(r['picks'])+' | Actual: '+(r['result'] or 'pending')+' | Review: '+r['review_on']+'\n'
    (PAPER/'three_picks.txt').write_text(text,encoding='utf-8')
    body='<meta charset="utf-8"><title>SC Pick 3 — three picks</title><style>body{font:18px system-ui;margin:35px;color:#163344}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:12px}strong{font-family:monospace;font-size:23px}</style><h1>Three picks per draw</h1><p>Noon means the midday draw; the second drawing is in the evening. Each pair of drawings is reviewed the next morning.</p><table><tr><th>Date</th><th>Draw</th><th>Predictions</th><th>Actual</th><th>Straight hit</th><th>Any-order hit</th></tr>'
    for r in rows:
        body+='<tr><td>'+r['date']+'</td><td>'+r['draw_type']+'</td><td><strong>'+', '.join(r['picks'])+'</strong></td><td>'+(r['result'] or 'Pending')+'</td><td>'+('Pending' if r['straight_hit'] is None else str(r['straight_hit']))+'</td><td>'+('Pending' if r['box_hit'] is None else str(r['box_hit']))+'</td></tr>'
    body+='</table><p>'+escape(summary['note'])+'</p><p><a href="index.html">Full paper study</a></p>'
    (PAPER/'three_picks.html').write_text(body,encoding='utf-8')
    print(text)

if __name__=='__main__':main()
