"""WEATHER-V1: isolated outdoor-weather research and prospective paper forecasts."""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from paper_track import ROOT, PAPER, TZ, deadline, write_once, digest

OUT = ROOT / 'weather'
VARIABLES = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure']
DIGITS = np.array([[int(c) for c in f'{i:03d}'] for i in range(1000)])

def fetch(url, path):
    if path.exists():
        return json.loads(path.read_text())
    with urllib.request.urlopen(url, timeout=90) as response:
        raw = response.read()
    value = json.loads(raw)
    if 'hourly' not in value:
        raise ValueError('Weather source omitted hourly data')
    path.write_bytes(raw)
    write_once(path.with_suffix('.meta.json'), {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'sha256': digest(path)})
    return value

def url(base, **extra):
    return base + '?' + urllib.parse.urlencode(dict(latitude=34.0007, longitude=-81.0348,
        hourly=','.join(VARIABLES), timezone='America/New_York', **extra))

def hourly_frame(value):
    f = pd.DataFrame(value['hourly'])
    f['date'] = f.time.str[:10]
    f['hour'] = f.time.str[11:13].astype(int)
    # 12:00 and 18:00 are fixed proxy hours near the 12:59 / 18:59 drawings.
    f = f[f.hour.isin([12, 18])].copy()
    f['draw_type'] = np.where(f.hour == 12, 'Day', 'Night')
    return f[['date', 'draw_type'] + VARIABLES]

def history():
    f = pd.read_csv(PAPER/'frozen/history.csv', dtype={'number': str})[['date','draw_type','number']]
    extra = [json.loads(p.read_text()) for p in (PAPER/'results').glob('*.json')]
    if extra:
        f = pd.concat([f, pd.DataFrame(extra)[['date','draw_type','number']]])
    return f.drop_duplicates(['date','draw_type']).sort_values(['date','draw_type'])

def features(frame, weather):
    t = pd.to_datetime(frame.date)
    angle = 2*np.pi*t.dt.dayofyear.to_numpy()/365.25
    x = np.column_stack([np.sin(angle), np.cos(angle), t.dt.dayofweek.to_numpy()/6])
    return np.column_stack([x, frame[VARIABLES].to_numpy(float)]) if weather else x

def probabilities(train, target, weather):
    x, z = features(train, weather), features(target, weather)
    y = np.array([[int(c) for c in n] for n in train.number])
    joint = np.ones((len(target), 1000))
    for j in range(3):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
        model.fit(x, y[:,j])
        # Reserve a small uniform mass for all digits, including unseen classes.
        p = np.full((len(target),10), 0.01)
        p[:,model[-1].classes_] += 0.9*model.predict_proba(z)
        joint *= p[:,DIGITS[:,j]]
    return joint/joint.sum(axis=1, keepdims=True)

def top3(p):
    return [f'{i:03d}' for i in np.argsort(-p, kind='stable')[:3]]

def initialize():
    for folder in ['sources','forecasts']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    protocol = OUT/'protocol.json'
    if not protocol.exists():
        write_once(protocol, {'version':'WEATHER-V1','created_utc':datetime.now(timezone.utc).isoformat(),
            'location':'Downtown Columbia outdoor grid proxy, 34.0007,-81.0348; not indoor studio measurements',
            'location_source':'https://www.sceducationlottery.com/Games/HowtoPlay',
            'weather_source':'https://open-meteo.com/en/docs/historical-weather-api',
            'training_start':'2020-01-01','features':VARIABLES,
            'method':'Separate Day/Night position logistic regressions; calendar sin/cos day of year and weekday; add temperature Celsius, humidity percent, pressure hPa. Training-only standardization, C=0.01, 10% digit uniform smoothing; independent-position product. Calendar-only control uses identical algorithm and rows.',
            'historical_test':'2023,2024,2025 calendar-year folds, train on earlier years only. ERA5 reanalysis is retrospective, not proof of pre-draw weather availability. Outdoor site continuity before 2020 not assumed.',
            'live':'Train on available historical weather through seven days before target or earlier; use forecast retrieved before registration cutoff. Freeze each record and source hash. Forecast/reanalysis domain mismatch is a limitation.',
            'endpoint':'200 verified prospective draws per stream; no interim promotion. Compare weather/calendar top3 exact hits and paired log loss, with multiplicity across streams/metrics; also show original baseline. No tickets.',
            'code_sha256':digest(Path(__file__))})
    p=json.loads(protocol.read_text())
    if p['code_sha256'] != digest(Path(__file__)):
        raise ValueError('Weather model code changed; create a new named version')

def historical_weather():
    end = (datetime.now(TZ).date()-timedelta(days=7)).isoformat()
    frames=[]
    for year in range(2020,int(end[:4])+1):
        stop=min(f'{year}-12-31',end)
        path=OUT/'sources'/f'era5_{year}_{stop}.json'
        frames.append(hourly_frame(fetch(url('https://archive-api.open-meteo.com/v1/archive',
            start_date=f'{year}-01-01',end_date=stop,models='era5'),path)))
    return pd.concat(frames,ignore_index=True)

def backtest(data):
    rows=[]
    for kind in ['Day','Night']:
        for year in [2023,2024,2025]:
            train=data[(data.draw_type==kind)&(data.date<f'{year}-01-01')]
            target=data[(data.draw_type==kind)&data.date.str.startswith(str(year))]
            if len(train)<500 or len(target)==0:
                raise ValueError('Insufficient historical coverage')
            actual=target.number.astype(int).to_numpy()
            for use in [False,True]:
                p=probabilities(train,target,use)
                hits=sum(n in top3(v) for n,v in zip(target.number,p))
                rows.append({'draw_type':kind,'year':year,'model':'weather' if use else 'calendar',
                    'draws':len(target),'top3_hits':hits,'log_loss':float(-np.log(p[np.arange(len(p)),actual]).mean())})
    write_once(OUT/'backtest.json',{'label':'Exploratory reanalysis association test, not an as-issued weather backtest','rows':rows})

def register_live(data):
    now=datetime.now(timezone.utc)
    targets=[]
    issues=[]
    for path in sorted((PAPER/'forecasts').glob('*.json')):
        f=json.loads(path.read_text())
        dest=OUT/'forecasts'/path.name
        if dest.exists() or f['date']<now.astimezone(TZ).date().isoformat():
            continue
        if now>=deadline(f['date'],f['draw_type']):
            issues.append(f"Missed weather cutoff: {path.stem}; not backfilled")
            continue
        targets.append((f,dest))
    if targets:
        source=OUT/'sources'/('forecast_'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        live=hourly_frame(fetch(url('https://api.open-meteo.com/v1/forecast',forecast_days=3),source))
        for f,dest in targets:
            target=live[(live.date==f['date'])&(live.draw_type==f['draw_type'])].dropna()
            cutoff=(datetime.fromisoformat(f['date']).date()-timedelta(days=7)).isoformat()
            train=data[(data.draw_type==f['draw_type'])&(data.date<=cutoff)]
            if len(target)!=1 or len(train)<500:
                issues.append('Missing weather coverage: '+dest.stem)
                continue
            wp=probabilities(train,target,True)[0]
            cp=probabilities(train,target,False)[0]
            registered=datetime.now(timezone.utc)
            if registered>=deadline(f['date'],f['draw_type']):
                issues.append('Missed weather cutoff after fitting: '+dest.stem)
                continue
            write_once(dest,{'version':'WEATHER-V1','date':f['date'],'draw_type':f['draw_type'],
                'registered_utc':registered.isoformat(),'training_through':train.date.max(),
                'training_sha256':hashlib.sha256(train.to_json(orient='records').encode()).hexdigest(),
                'weather_source':str(source.relative_to(OUT)),'weather_sha256':digest(source),
                'outdoor_forecast':target[VARIABLES].iloc[0].to_dict(),
                'weather_picks':top3(wp),'calendar_picks':top3(cp),'baseline_picks':f['model_top10'][:3],
                'weather_probabilities':wp.tolist(),'calendar_probabilities':cp.tolist()})
    return issues

def review(issues):
    rows=[]
    for path in sorted((OUT/'forecasts').glob('*.json')):
        f=json.loads(path.read_text())
        if datetime.fromisoformat(f['registered_utc'])>=deadline(f['date'],f['draw_type']):
            raise ValueError('Late weather forecast')
        source=OUT/Path(f['weather_source'].replace('\\','/'))
        if digest(source)!=f['weather_sha256']:
            raise ValueError('Changed weather snapshot')
        result=PAPER/'results'/path.name
        actual=json.loads(result.read_text())['number'] if result.exists() else None
        row={k:f[k] for k in ['date','draw_type','weather_picks','calendar_picks','baseline_picks','outdoor_forecast']}
        row['actual']=actual
        for model in ['weather','calendar','baseline']:
            row[model+'_hit']=actual in f[model+'_picks'] if actual is not None else None
        for model in ['weather','calendar']:
            row[model+'_log_loss']=float(-np.log(f[model+'_probabilities'][int(actual)])) if actual is not None else None
        rows.append(row)
    (OUT/'summary.json').write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),'issues':issues,'rows':rows},indent=2))
    text='# Weather experiment — WEATHER-V1\n\nOutdoor Columbia temperature, humidity and pressure; not studio measurements. Experimental picks supplement the original baseline. Historical reanalysis is exploratory; no predictive advantage is established.\n\n'
    text+='| Date | Drawing | Weather picks | Calendar control | Original baseline | Actual |\n|---|---|---|---|---|---|\n'
    for r in rows:
        text+=f"| {r['date']} | {r['draw_type']} | {', '.join(r['weather_picks'])} | {', '.join(r['calendar_picks'])} | {', '.join(r['baseline_picks'])} | {r['actual'] or 'Pending'} |\n"
    text+='\nIssues: '+('; '.join(issues) or 'None')+'\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')
    print(text)

def main():
    initialize()
    try:
        data=history().merge(historical_weather(),on=['date','draw_type'],how='inner')
        missing=int(data[VARIABLES].isna().any(axis=1).sum())
        data=data.dropna(subset=VARIABLES)
        if not (OUT/'backtest.json').exists(): backtest(data)
        issues=register_live(data)
        if missing: issues.append(f'{missing} rows excluded for missing historical weather')
        review(issues)
    except Exception as error:
        review([f'Weather experiment failed: {type(error).__name__}: {error}'])
        raise

if __name__=='__main__': main()
