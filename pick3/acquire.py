"""Download public SC Pick 3 yearly archives, retaining original pages and hashes."""
import argparse, hashlib, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

def parse_archive(body):
    soup = BeautifulSoup(body, 'html.parser')
    records = []
    for tr in soup.select('tr'):
        cells = tr.find_all('td',recursive=False)
        if len(cells) != 3:
            continue
        try:
            d = datetime.strptime(cells[0].get_text(' ',strip=True),'%A, %B %d, %Y').date().isoformat()
        except ValueError:
            continue
        for cell,kind in zip(cells[1:],('Day','Night')):
            balls = cell.select('[class*="number-part-"]')
            if not balls:
                continue
            digits = [x.get_text(strip=True) for x in balls]
            assert len(digits) == 3 and all(re.fullmatch('[0-9]',x) for x in digits), (d,digits)
            fireball = cell.select_one('.fireball')
            records.append({'date':d,'draw_type':kind,'number':''.join(digits),
                            'fireball':fireball.get_text(strip=True) if fireball else ''})
    return records

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=Path(__file__).parent)
    args = parser.parse_args(); raw = args.out/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
    def fetch(year):
        url = f'https://sc.pick-3.com/numbers/{year}'
        p = raw/f'archive_{year}.html'
        if not p.exists() or year == datetime.now().year:
            p.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        body = p.read_bytes(); rows = parse_archive(body)
        assert rows and all(r['date'].startswith(str(year)) for r in rows)
        meta = {'url':url,'file':p.name,'sha256':hashlib.sha256(body).hexdigest(),'drawings':len(rows),
                'first':min(r['date'] for r in rows),'last':max(r['date'] for r in rows)}
        print(year,len(rows),flush=True)
        return rows,meta
    rows,meta = [],[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for records,source in pool.map(fetch,range(2002,datetime.now().year+1)):
            rows.extend(records);meta.append(source)
    f = pd.DataFrame(rows).sort_values(['date','draw_type'])
    assert not f.duplicated(['date','draw_type']).any()
    f.to_csv(args.out/'data/sc_pick3_archive.csv',index=False)
    manifest = {'retrieved_utc':datetime.now(timezone.utc).isoformat(),'primary_archive':'https://sc.pick-3.com/numbers/2026',
                'source_type':'Third-party archive; official cross-check recorded separately',
                'sources':meta,'drawings':len(f),'coverage':f.groupby('draw_type').date.agg(['min','max','count']).to_dict('index')}
    (raw/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['coverage'],indent=2))

if __name__ == '__main__': main()
