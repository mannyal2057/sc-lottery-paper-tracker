"""Verify completed seven-day blocks; never rewrite the registered experiment."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from collections import Counter
import json,hashlib
from bs4 import BeautifulSoup
from paper_track import PAPER,TZ,deadline,stored_path,check_frozen,journal

def main():
    protocol=json.loads((PAPER/'protocol.json').read_text())
    check_frozen(protocol);journal()
    top3=json.loads((PAPER/'three_pick_protocol.json').read_text())
    selection_created=datetime.fromisoformat(top3['created_utc'])
    start=min(datetime.fromisoformat(x).date() for x in protocol['initial_forecast_date'].values())
    today=datetime.now(TZ).date();completed=(today-start).days//7
    folder=PAPER/'weekly';folder.mkdir(exist_ok=True)
    if completed<1:
        state={'status':'COLLECTING','first_review_date':str(start+timedelta(days=7)),
               'first_window_start':str(start),'first_window_end':str(start+timedelta(days=6))}
        (folder/'status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(json.dumps(state,indent=2));return
    first=start+timedelta(days=7*(completed-1));last=first+timedelta(days=6)
    rows=[];issues=[]
    for path in sorted((PAPER/'forecasts').glob('*.json')):
        f=json.loads(path.read_text());date=datetime.fromisoformat(f['date']).date()
        if not first<=date<=last:continue
        if selection_created>=deadline(f['date'],f['draw_type']):continue
        row={'date':f['date'],'draw_type':f['draw_type'],'picks':f['model_top10'][:3],
             'random_picks':f['random_top10'][:3],'status':'pending'}
        if datetime.fromisoformat(f['registered_utc'])>=deadline(f['date'],f['draw_type']):
            raise ValueError('Late registered prediction: '+path.name)
        result_path=PAPER/'results'/path.name
        if result_path.exists():
            r=json.loads(result_path.read_text());snapshot=stored_path(r['official_snapshot'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['forecast_sha256']
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==r['official_snapshot_sha256']
            soup=BeautifulSoup(snapshot.read_bytes(),'html.parser');observed={}
            for box in soup.select('.drawResultsaccordion-title'):
                d=datetime.strptime(box.select_one('.lightblue-bg').get_text(strip=True),'%B %d, %Y').date().isoformat()
                k={'Midday':'Day','Evening':'Night'}[box.select_one('.lightblue-bg-right').get_text(strip=True)]
                observed[(d,k)]=''.join(x.get_text(strip=True) for x in box.select('li.number:not(.fireball)'))
            assert observed.get((f['date'],f['draw_type']))==r['number'],'Official snapshot mismatch'
            actual=r['number']
            row.update({'status':'verified','actual':actual,'straight_hit':actual in row['picks'],
                'random_hit':actual in row['random_picks'],
                'box_hit':any(sorted(p)==sorted(actual) for p in row['picks']),
                'best_position_matches':max(sum(a==b for a,b in zip(p,actual)) for p in row['picks']),
                'best_digit_overlap':max(sum((Counter(p)&Counter(actual)).values()) for p in row['picks'])})
        else:issues.append(f['date']+' '+f['draw_type']+': result pending/unverified')
        rows.append(row)
    expected=[]
    for day in (first+timedelta(days=i) for i in range(7)):
        for kind in ('Day','Night'):
            if kind=='Day' and (day.weekday()==6 or (day.month,day.day)==(12,25)):continue
            expected.append((str(day),kind))
    keys={(r['date'],r['draw_type']) for r in rows}
    for date,kind in expected:
        if (date,kind) not in keys:issues.append(date+' '+kind+': no eligible pre-draw prediction; not backfilled')
    totals={}
    for kind in ('Day','Night'):
        scored=[r for r in rows if r['draw_type']==kind and r['status']=='verified']
        hits=sum(r['straight_hit'] for r in scored);random=sum(r['random_hit'] for r in scored)
        totals[kind]={'verified':len(scored),'model_top3_hits':hits,'random_top3_hits':random,
            'box_hits':sum(r['box_hit'] for r in scored),'model_paper_net_usd':500*hits-3*len(scored),
            'random_paper_net_usd':500*random-3*len(scored)}
    report={'start':str(first),'end':str(last),'reviewed_utc':datetime.now(timezone.utc).isoformat(),
        'version':'TOP3-V1','totals':totals,'rows':rows,'issues':issues,
        'decision':'Keep the frozen baseline. A week is too small to establish a betting edge. Any proposed adjustment must be documented as a new prospective version and compared alongside the baseline, not substituted into past records.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=folder/f'{first}_to_{last}_{stamp}.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    text=f'# Weekly review: {first} through {last}\n\n'+report['decision']+'\n\n'
    for kind,t in totals.items():text+=f"{kind}: {t['verified']} verified draws; model top-three hits {t['model_top3_hits']}; random top-three hits {t['random_top3_hits']}; any-order hits {t['box_hits']}.\n\n"
    text+='Issues: '+('; '.join(issues) if issues else 'None')+'\n'
    (folder/'latest_review.md').write_text(text,encoding='utf-8')
    (folder/'status.json').write_text(json.dumps({'status':'REVIEWED','report_file':path.name,'start':str(first),'end':str(last),'issues':issues},indent=2),encoding='utf-8')
    print(text)

if __name__=='__main__':main()
