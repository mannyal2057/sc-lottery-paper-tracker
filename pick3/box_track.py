"""BOX-V1: three distinct box classes ranked by summed baseline probability."""
import hashlib,json
from datetime import datetime,timezone
import numpy as np
from paper_track import ROOT,PAPER,deadline,write_once,digest,check_frozen,journal

OUT=ROOT/'box'
GROUPS={}
for n in range(1000):GROUPS.setdefault(''.join(sorted(f'{n:03d}')),[]).append(n)
GROUPS={k:v for k,v in GROUPS.items() if len(v) in (3,6)}

def select(probabilities):
    p=np.asarray(probabilities,float)
    if p.shape!=(1000,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
        raise ValueError('Invalid baseline probabilities')
    order=sorted(GROUPS,key=lambda k:(-p[GROUPS[k]].sum(),k))[:3]
    return [{'digits':k,'ways':len(GROUPS[k]),'model_probability':float(p[GROUPS[k]].sum()),
             'gross_prize_usd':160 if len(GROUPS[k])==3 else 80} for k in order]

def payout(picks,actual):
    key=''.join(sorted(actual))
    return sum(x['gross_prize_usd'] for x in picks if x['digits']==key)

def main():
    for name in ['forecasts','results']:(OUT/name).mkdir(parents=True,exist_ok=True)
    check_frozen(json.loads((PAPER/'protocol.json').read_text()));journal()
    protocol=OUT/'protocol.json'
    if not protocol.exists():
        write_once(protocol,{'version':'BOX-V1','created_utc':datetime.now(timezone.utc).isoformat(),
            'selection':'Sum frozen baseline probabilities over each unordered digit multiset. Exclude triples (not a 3-way or 6-way Box). Rank classes by total probability, canonical digits as tie break. Choose three distinct classes; never count permutations as separate boxes.',
            'cost_per_draw_usd':3,'cost_per_box_usd':1,'gross_prize_3way_usd':160,'gross_prize_6way_usd':80,
            'rules_source':'https://www.sceducationlottery.com/documents/games/onlinegames/GameRules_Pick3.pdf',
            'control':'Three random distinct box classes with matching 3-way/6-way composition. Equal coverage and cost.',
            'registration':'Only before original cutoff and after this protocol exists. Never backfill or overwrite. Paper only, no tickets or Fireball.',
            'evaluation':'Score only baseline-verified results. Report exact box hits, coverage-specific chance, gross winnings and net after cost. More coverage is not a predictive advantage; compare with matched random boxes. Weekly results descriptive. Original study endpoint remains in force; report shorter Box sample at that point.',
            'code_sha256':digest(__file_path())})
    if json.loads(protocol.read_text())['code_sha256']!=digest(__file_path()):raise ValueError('Box code changed; new version required')
    issues=[]
    for path in sorted((PAPER/'forecasts').glob('*.json')):
        f=json.loads(path.read_text());dest=OUT/'forecasts'/path.name
        now=datetime.now(timezone.utc)
        if not dest.exists() and now<deadline(f['date'],f['draw_type']):
            picks=select(f['probabilities'])
            seed=int(hashlib.sha256(('BOX-V1|'+path.stem).encode()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed);random=[];used=set()
            for x in picks:
                eligible=[k for k,v in GROUPS.items() if len(v)==x['ways'] and k not in used]
                k=str(rng.choice(eligible));used.add(k)
                random.append({'digits':k,'ways':x['ways'],'gross_prize_usd':x['gross_prize_usd']})
            registered=datetime.now(timezone.utc)
            if registered>=deadline(f['date'],f['draw_type']):continue
            write_once(dest,{'version':'BOX-V1','date':f['date'],'draw_type':f['draw_type'],
                'registered_utc':registered.isoformat(),'baseline_sha256':digest(path),'picks':picks,
                'random_picks':random,'cost_usd':3,'fair_hit_probability':sum(x['ways'] for x in picks)/1000})
        if not dest.exists():continue
        b=json.loads(dest.read_text())
        if digest(path)!=b['baseline_sha256']:raise ValueError('Baseline forecast changed')
        if datetime.fromisoformat(b['registered_utc'])>=deadline(b['date'],b['draw_type']):raise ValueError('Late box forecast')
        result=PAPER/'results'/path.name;scored=OUT/'results'/path.name
        if result.exists() and not scored.exists():
            actual=json.loads(result.read_text())['number']
            gross=payout(b['picks'],actual);rgross=payout(b['random_picks'],actual)
            write_once(scored,{'actual':actual,'forecast_sha256':digest(dest),'verified_result_sha256':digest(result),
                'gross_usd':gross,'net_usd':gross-3,'random_gross_usd':rgross,'random_net_usd':rgross-3})
        if scored.exists():
            r=json.loads(scored.read_text())
            if r['forecast_sha256']!=digest(dest) or r['verified_result_sha256']!=digest(result):raise ValueError('Box audit mismatch')
    rows=[]
    for path in sorted((OUT/'forecasts').glob('*.json')):
        b=json.loads(path.read_text());r=OUT/'results'/path.name
        b['result']=json.loads(r.read_text()) if r.exists() else None;rows.append(b)
    (OUT/'summary.json').write_text(json.dumps({'rows':rows,'issues':issues},indent=2))
    text='# Three Box selections per drawing — BOX-V1\n\nPaper only. Each selection costs $1 hypothetically; $3 per drawing, separately from straight tests. Any order wins. 3-way gross prize $160; 6-way gross prize $80. No Fireball. Selections maximize model hit probability, not expected payout.\n\n'
    text+='| Date | Drawing | Three boxes (ways) | Fair hit chance | Actual | Box net |\n|---|---|---|---:|---|---|\n'
    for b in rows:
        r=b['result'];label=', '.join(f"{x['digits']} ({x['ways']}-way)" for x in b['picks'])
        text+=f"| {b['date']} | {b['draw_type']} | {label} | {b['fair_hit_probability']:.1%} | {r['actual'] if r else 'Pending'} | {r['net_usd'] if r else 'Pending'} |\n"
    text+='\nHigher hit probability comes with lower prizes. Random controls have the same cost and permutation coverage. Old straight any-order observations are not retroactively Box wagers.\n'
    (OUT/'README.md').write_text(text,encoding='utf-8');print(text)

def __file_path():
    from pathlib import Path
    return Path(__file__)

if __name__=='__main__':main()
