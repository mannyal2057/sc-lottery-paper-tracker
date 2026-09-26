"""Prospective paper study: exclusive-create records, fixed model, no backfilled picks."""
import argparse,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from html import escape
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT=Path(__file__).parent
PAPER=ROOT/'paper'
TZ=ZoneInfo('America/New_York')

def utcnow():return datetime.now(timezone.utc)
def public_page(url):
    """Avoid cached pre-draw pages when checking newly published results."""
    separator='&' if '?' in url else '?'
    request=urllib.request.Request(url+separator+'paper_check='+str(int(utcnow().timestamp())),
        headers={'Cache-Control':'no-cache','Pragma':'no-cache'})
    with urllib.request.urlopen(request,timeout=30) as response:
        return response.read()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stored_path(name):
    """Read legacy Windows journal paths without rewriting signed record bytes."""
    return PAPER/Path(name.replace('\\','/'))
def write_once(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)
def next_day(date,kind):
    d=datetime.strptime(date,'%Y-%m-%d').date()+timedelta(days=1)
    while kind=='Day' and (d.weekday()==6 or (d.month,d.day)==(12,25)):d+=timedelta(days=1)
    return d.isoformat()
def deadline(date,kind):
    return datetime.fromisoformat(date+('T12:45:00' if kind=='Day' else 'T18:45:00')).replace(tzinfo=TZ)
def random_picks(date,kind):
    seed=int(hashlib.sha256(('SC-PAPER-V1|'+date+'|'+kind).encode()).hexdigest()[:16],16)
    return [f'{i:03d}' for i in np.random.default_rng(seed).choice(1000,10,replace=False)]

def register(folder,date,kind,p,as_of,now=None):
    now=now or utcnow()
    if now>=deadline(date,kind):raise ValueError('Cannot register at or after pre-draw cutoff')
    if as_of>=date:raise ValueError('Training cutoff must precede target date')
    p=np.asarray(p,float)
    if p.shape!=(1000,) or not np.isfinite(p).all() or np.any(p<=0) or not np.isclose(p.sum(),1):
        raise ValueError('Invalid probability vector')
    order=np.lexsort((np.random.default_rng(20260909).random(1000),-p))
    obj={'study':'SC-PAPER-V1','date':date,'draw_type':kind,'registered_utc':now.isoformat(),
         'deadline_local':deadline(date,kind).isoformat(),'training_through':as_of,
         'model_top10':[f'{i:03d}' for i in order[:10]],'random_top10':random_picks(date,kind),
         'probabilities':p.tolist(),'paper_cost_each_usd':10,'gross_straight_prize_usd':500}
    path=folder/f'{date}_{kind}.json'
    write_once(path,obj)
    return path

def initialize():
    for name in ('forecasts','results','sources','frozen/pick3'):(PAPER/name).mkdir(parents=True,exist_ok=True)
    if (PAPER/'protocol.json').exists():return
    meta=json.loads((ROOT/'models/model_results.json').read_text())
    for p in (ROOT/'pick3').glob('*.py'):shutil.copy2(p,PAPER/'frozen/pick3'/p.name)
    shutil.copy2(ROOT/'data/sc_pick3_clean.csv',PAPER/'frozen/history.csv')
    for kind in ('Day','Night'):shutil.copy2(ROOT/f'reports/{kind}_component_probabilities.csv',PAPER/f'frozen/{kind}_probabilities.csv')
    protocol={'study':'SC-PAPER-V1','created_utc':utcnow().isoformat(),
        'weights':{k:meta[k]['static_weights_from_validation_only'] for k in ('Day','Night')},
        'endpoint_draws_per_stream':200,'initial_forecast_date':{k:meta[k]['forecast_date'] for k in ('Day','Night')},
        'rules':'Register before 12:45 Day / 18:45 Night America/New_York. Never backfill missed forecasts or replace registered picks. Ten distinct model and random straights; $10 hypothetical cost each. Fixed mixture weights and model code, refit only on preceding same-type draws. No tickets purchased.',
        'inference':'Only after both streams complete 200 scored draws: four Holm-corrected tests, Day/Night model top10 vs .01 and paired model vs random exact discordant tests. Night anomaly: first 200 Night outcomes, test excess tens=0 and deficit tens=7 using one-sided binomial tests with Bonferroni factor 2. Interim rates descriptive; 200 draws has limited power for rare hits.',
        'integrity_note':'Exclusive-create JSON records plus SHA256 journal are locally auditable; not an externally notarized or tamper-proof timestamp service.',
        'frozen_hashes':{str(p.relative_to(PAPER)):digest(p) for p in (PAPER/'frozen').rglob('*') if p.is_file()}}
    write_once(PAPER/'protocol.json',protocol)

def check_frozen(protocol):
    for name,sha in protocol['frozen_hashes'].items():
        if digest(stored_path(name))!=sha:raise ValueError('Frozen input/model changed: '+name)

def journal():
    path=PAPER/'audit.jsonl'
    previous='0'*64;known={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line); expected=row.pop('chain_hash')
            if row['previous']!=previous or hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()!=expected:
                raise ValueError('Audit chain mismatch')
            if digest(stored_path(row['file']))!=row['sha256']:raise ValueError('Registered record changed: '+row['file'])
            previous=expected;known[row['file'].replace('\\','/')]=True
    for sub in ('forecasts','results'):
        for p in sorted((PAPER/sub).glob('*.json')):
            name=p.relative_to(PAPER).as_posix()
            if name in known:continue
            row={'file':name,'sha256':digest(p),'previous':previous,'logged_utc':utcnow().isoformat()}
            previous=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();row['chain_hash']=previous
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')

def fetch():
    from acquire import parse_archive
    stamp=utcnow().strftime('%Y%m%dT%H%M%S%fZ')
    official_url='https://sceducationlottery.com/Games/Pick3'
    body=public_page(official_url)
    soup=BeautifulSoup(body,'html.parser');official={}
    for box in soup.select('.drawResultsaccordion-title'):
        date=datetime.strptime(box.select_one('.lightblue-bg').get_text(strip=True),'%B %d, %Y').date().isoformat()
        kind={'Midday':'Day','Evening':'Night'}[box.select_one('.lightblue-bg-right').get_text(strip=True)]
        number=''.join(x.get_text(strip=True) for x in box.select('li.number:not(.fireball)'))
        assert len(number)==3 and number.isdigit()
        official[(date,kind)]=number
    if not official:raise ValueError('Official source did not parse')
    state_hash=hashlib.sha256(json.dumps(sorted((d,k,n) for (d,k),n in official.items())).encode()).hexdigest()[:20]
    official_path=PAPER/'sources'/f'{state_hash}_official.html'
    if not official_path.exists():official_path.write_bytes(body)
    years={utcnow().astimezone(TZ).year}
    years.update(int(p.name[:4]) for p in (PAPER/'forecasts').glob('*.json') if not (PAPER/'results'/p.name).exists())
    records=[]
    for year in sorted(years):
        body=public_page(f'https://sc.pick-3.com/numbers/{year}')
        parsed=parse_archive(body)
        if not parsed:raise ValueError('Archive source did not parse')
        state_hash=hashlib.sha256(json.dumps(parsed,sort_keys=True).encode()).hexdigest()[:20]
        snapshot=PAPER/'sources'/f'{state_hash}_archive_{year}.html'
        if not snapshot.exists():snapshot.write_bytes(body)
        records.extend(parsed)
    archive={(r['date'],r['draw_type']):r['number'] for r in records}
    for key,value in official.items():
        if key in archive and archive[key]!=value:raise ValueError('Sources disagree at '+str(key))
    pending=[]
    for p in (PAPER/'forecasts').glob('*.json'):
        f=json.loads(p.read_text());key=(f['date'],f['draw_type'])
        if key in official and key not in archive:
            pending.append({'date':key[0],'draw_type':key[1],'official_number':official[key],
                            'status':'Awaiting archive confirmation; not scored',
                            'official_snapshot':str(official_path.relative_to(PAPER))})
    (PAPER/'source_status.json').write_text(json.dumps({'checked_utc':utcnow().isoformat(),
        'awaiting_archive_confirmation':pending},indent=2),encoding='utf-8')
    return archive,official,str(official_path.relative_to(PAPER))

def render(protocol):
    results=[json.loads(p.read_text()) for p in sorted((PAPER/'results').glob('*.json'))]
    forecasts=[json.loads(p.read_text()) for p in sorted((PAPER/'forecasts').glob('*.json'))]
    records=[]
    for x in forecasts:
        y=next((r for r in results if r['date']==x['date'] and r['draw_type']==x['draw_type']),None)
        records.append({'date':x['date'],'draw':x['draw_type'],'registered_utc':x['registered_utc'],
            'model_picks':', '.join(x['model_top10']),'random_picks':', '.join(x['random_top10']),
            'result':y['number'] if y else 'pending','model_hit':y['model_hit'] if y else '',
            'random_hit':y['random_hit'] if y else ''})
    totals={}
    for k in ('Day','Night'):
        r=[x for x in results if x['draw_type']==k]
        totals[k]={'scored':len(r),'model_hits':sum(x['model_hit'] for x in r),'random_hits':sum(x['random_hit'] for x in r)}
        totals[k]['model_paper_net_usd']=500*totals[k]['model_hits']-10*len(r)
        totals[k]['random_paper_net_usd']=500*totals[k]['random_hits']-10*len(r)
    night=[r for r in results if r['draw_type']=='Night'][:200]
    anomaly={'night_draws':len(night),'tens_zero':sum(r['number'][1]=='0' for r in night),'tens_seven':sum(r['number'][1]=='7' for r in night)}
    from scipy.stats import binomtest
    inference={}
    if len(night)==200:
        inference['anomaly_zero_p_bonferroni']=min(1,2*binomtest(anomaly['tens_zero'],200,.1,alternative='greater').pvalue)
        inference['anomaly_seven_p_bonferroni']=min(1,2*binomtest(anomaly['tens_seven'],200,.1,alternative='less').pvalue)
    if all(totals[k]['scored']>=200 for k in totals):
        from pick3.backtest import holm
        pvalues=[];names=[]
        for k in ('Day','Night'):
            r=[x for x in results if x['draw_type']==k][:200]
            pvalues.append(binomtest(sum(x['model_hit'] for x in r),200,.01,alternative='greater').pvalue);names.append(k+'_vs_uniform')
            a=sum(x['model_hit'] and not x['random_hit'] for x in r);b=sum(x['random_hit'] and not x['model_hit'] for x in r)
            pvalues.append(binomtest(a,a+b,.5,alternative='greater').pvalue if a+b else 1);names.append(k+'_vs_random')
        inference.update(dict(zip(names,holm(pvalues))))
    summary={'updated_utc':utcnow().isoformat(),'totals':totals,'anomaly':anomaly,'endpoint_tests':inference,
             'registered':len(forecasts),'pending':len(forecasts)-len(results),'complete':all(totals[k]['scored']>=200 for k in totals)}
    scored_keys={(r['date'],r['draw_type']) for r in results}
    summary['overdue_unverified']=[x['date']+' '+x['draw_type'] for x in forecasts if (x['date'],x['draw_type']) not in scored_keys and utcnow()>deadline(x['date'],x['draw_type'])+timedelta(days=2)]
    source_status=PAPER/'source_status.json'
    summary['awaiting_archive_confirmation']=json.loads(source_status.read_text()).get('awaiting_archive_confirmation',[]) if source_status.exists() else []
    (PAPER/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    table=pd.DataFrame(records);table.to_csv(PAPER/'tracking.csv',index=False)
    html='<meta charset="utf-8"><title>SC Pick 3 paper tracking</title><style>body{font:16px system-ui;margin:35px;color:#173448}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>Prospective paper tracking</h1><p>No ticket purchases. Predictions are registered before draw cutoffs. Pending results are not losses. Interim results are descriptive; formal review after 200 scored draws per stream.</p><h2>Scoreboard</h2>'+pd.DataFrame(totals).T.to_html()+'<h2>Night tens anomaly</h2><pre>'+escape(json.dumps(anomaly,indent=2))+'</pre><h2>Registered draws</h2>'+table.to_html(index=False)+'<p>Raw forecast/result records, source snapshots and the hash journal are stored alongside this report. Local timestamps are not externally notarized.</p>'
    (PAPER/'index.html').write_text(html,encoding='utf-8')
    return summary

def main():
    initialize();protocol=json.loads((PAPER/'protocol.json').read_text());check_frozen(protocol);journal()
    # Register the existing forecast NOW, not after retrieving its outcome.
    if not list((PAPER/'forecasts').glob('*.json')):
        for kind,date in protocol['initial_forecast_date'].items():
            if utcnow()<deadline(date,kind):
                p=pd.read_csv(PAPER/f'frozen/{kind}_probabilities.csv').ensemble_static.to_numpy()
                base=pd.read_csv(PAPER/'frozen/history.csv',dtype=str);last=base[base.draw_type==kind].date.max()
                register(PAPER/'forecasts',date,kind,p,last)
        journal()
    archive,official,source=fetch()
    for path in sorted((PAPER/'forecasts').glob('*.json')):
        result_path=PAPER/'results'/path.name
        if result_path.exists():continue
        forecast=json.loads(path.read_text());key=(forecast['date'],forecast['draw_type'])
        if key not in official or key not in archive:continue
        if utcnow()<=deadline(*key)+timedelta(minutes=14):raise ValueError('Source exposed a result before draw time')
        number=official[key]
        result={'date':key[0],'draw_type':key[1],'number':number,'scored_utc':utcnow().isoformat(),
            'official_snapshot':source,'official_snapshot_sha256':digest(PAPER/source),
            'forecast_sha256':digest(path),'model_hit':number in forecast['model_top10'],
            'random_hit':number in forecast['random_top10'],
            'model_log_loss':float(-np.log(forecast['probabilities'][int(number)]))}
        write_once(result_path,result)
    journal()
    base=pd.read_csv(PAPER/'frozen/history.csv',dtype={'number':str})
    for kind in ('Day','Night'):
        if len(list((PAPER/'forecasts').glob('*_'+kind+'.json')))>=200:continue
        g=base[base.draw_type==kind][['date','draw_type','number']].copy()
        added=[{'date':d,'draw_type':k,'number':v} for (d,k),v in archive.items() if k==kind and d>g.date.max()]
        if added:g=pd.concat([g,pd.DataFrame(added)],ignore_index=True).sort_values('date')
        last=g.date.max();target=next_day(last,kind)
        if (PAPER/'forecasts'/f'{target}_{kind}.json').exists() or utcnow()>=deadline(target,kind):continue
        if (last,kind) not in official or (last,kind) not in archive:continue
        if utcnow()<=deadline(last,kind)+timedelta(minutes=14):raise ValueError('History includes an uncompleted draw')
        # Only this stream's preceding history is used. Frozen code and weights stay fixed.
        sys.path.insert(0,str(PAPER/'frozen'))
        from pick3.models import basic,MLModels,feature_matrix,normalize
        h=np.array([[int(x) for x in n] for n in g.number])
        p=basic(h);X=feature_matrix(h);engine=MLModels();engine.fit(X,h,len(h));p.update(engine.predict(X[-1],len(h)))
        combined=normalize(sum(w*p[name] for name,w in protocol['weights'][kind].items()))
        register(PAPER/'forecasts',target,kind,combined,last)
        journal()
    print(json.dumps(render(protocol),indent=2))

if __name__=='__main__':main()
