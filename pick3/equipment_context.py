"""Validate sourced equipment IDs; unknown IDs never become predictive categories."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from paper_track import ROOT, PAPER, deadline

OUT = ROOT / 'equipment'

def validate(record, root=OUT):
    if record['draw_type'] not in ('Day','Night') or record['position'] not in ('H','T','U'):
        raise ValueError('Invalid draw type or digit position')
    datetime.strptime(record['date'],'%Y-%m-%d')
    for field in ['machine_id','ball_set_id','source_url']:
        if not isinstance(record.get(field),str) or not record[field].strip():
            raise ValueError('Missing '+field)
    if not record['source_url'].startswith('https://'):
        raise ValueError('Expected public HTTPS source URL')
    source=(root/record['source_file']).resolve()
    if not source.is_relative_to(root.resolve()) or not source.is_file():
        raise ValueError('Source snapshot must be inside equipment folder')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_sha256']:
        raise ValueError('Source snapshot hash mismatch')
    times=[datetime.fromisoformat(record[k]) for k in ('public_available_utc','retrieved_utc')]
    if any(t.tzinfo is None for t in times):
        raise ValueError('Source times require timezone')
    if times[0]>times[1]:
        raise ValueError('Public availability cannot follow retrieval')
    if record.get('mapping_verified') is not True:
        raise ValueError('Source-to-drawing ID mapping has not been verified')
    return max(times)<deadline(record['date'],record['draw_type'])

def load_records(root=OUT):
    records={}
    for path in sorted((root/'records').glob('*.json')):
        r=json.loads(path.read_text(encoding='utf-8'))
        eligible=validate(r,root)
        key=(r['date'],r['draw_type'],r['position'])
        if key in records:
            raise ValueError('Duplicate equipment mapping: '+str(key))
        records[key]={**r,'captured_before_cutoff':eligible}
    return records

def main():
    (OUT/'records').mkdir(parents=True,exist_ok=True)
    (OUT/'sources').mkdir(exist_ok=True)
    records=load_records()
    rows=[]; groups=defaultdict(lambda:[0]*10)
    for path in sorted((PAPER/'forecasts').glob('*.json')):
        f=json.loads(path.read_text())
        rpath=PAPER/'results'/path.name
        result=json.loads(rpath.read_text())['number'] if rpath.exists() else None
        positions={}
        for j,pos in enumerate('HTU'):
            record=records.get((f['date'],f['draw_type'],pos))
            positions[pos]=None if record is None else {k:record[k] for k in ('machine_id','ball_set_id','captured_before_cutoff')}
            if result is not None and record is not None:
                groups[(f['draw_type'],pos,record['machine_id'],record['ball_set_id'])][int(result[j])]+=1
        rows.append({'date':f['date'],'draw_type':f['draw_type'],'positions':positions,
            'eligible_for_future_equipment_model':all(v and v['captured_before_cutoff'] for v in positions.values())})
    stats=[{'draw_type':k[0],'position':k[1],'machine_id':k[2],'ball_set_id':k[3],
            'digit_counts':v,'draws':sum(v)} for k,v in groups.items()]
    obj={'updated_utc':datetime.now(timezone.utc).isoformat(),
        'status':'AWAITING_SOURCED_IDS' if not records else 'DESCRIPTIVE_ONLY',
        'verified_mappings':len(records),'rows':rows,'equipment_digit_counts':stats,
        'note':'No equipment prediction model activated. Unknown IDs stay null; post-draw records are descriptive only. Coverage here is the prospective paper study.'}
    (OUT/'summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in obj.items() if k not in ('rows','equipment_digit_counts')},indent=2))

if __name__=='__main__':main()
