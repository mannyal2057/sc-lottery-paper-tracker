from pathlib import Path
from datetime import date
import hashlib
import re
import pandas as pd

COLUMNS = ['date', 'draw_type', 'number', 'H', 'T', 'U']
ALIASES = {'draw_date': 'date', 'drawing_date': 'date', 'draw': 'draw_type',
           'drawing': 'draw_type', 'draw_time': 'draw_type', 'winning_number': 'number',
           'winning_numbers': 'number', 'result': 'number'}
TYPES = {'day': 'Day', 'midday': 'Day', 'afternoon': 'Day',
         'night': 'Night', 'evening': 'Night', 'eve': 'Night'}

def discover(root):
    """Inventory only: game identity must be confirmed by the supplied source."""
    return sorted(str(p) for p in Path(root).rglob('*') if p.is_file()
                  and p.suffix.lower() in {'.csv', '.xlsx', '.xls', '.json'}
                  and not any(x in p.parts for x in ('reports', 'models', '__pycache__')))

def read(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, dtype=str)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        return pd.read_excel(path, dtype=str)
    if path.suffix.lower() == '.json':
        return pd.read_json(path, dtype=False).astype(str)
    raise ValueError('Supported formats: CSV, XLSX, XLS, or JSON array of records')

def clean(frame, today=None):
    today = today or date.today()
    f = frame.copy()
    f.columns = [ALIASES.get(re.sub(r'\W+', '_', str(c).strip().lower()),
                           re.sub(r'\W+', '_', str(c).strip().lower())) for c in f.columns]
    if f.columns.duplicated().any():
        raise ValueError('Ambiguous duplicate column names after normalization')
    if not {'date', 'draw_type', 'number'} <= set(f):
        raise ValueError('Required columns: date, draw_type, number; draw type is never guessed')
    out, rejected = [], []
    for idx, row in f.iterrows():
        try:
            rawdate = str(row['date']).strip()
            # Require unambiguous dates; do not guess day/month ordering.
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[ T]00:00:00)?', rawdate):
                raise ValueError('date must be ISO YYYY-MM-DD')
            d = date.fromisoformat(rawdate[:10])
            if d > today:
                raise ValueError('future date')
            kind = TYPES.get(str(row['draw_type']).strip().lower())
            if kind is None:
                raise ValueError('unknown draw type')
            n = str(row['number']).strip()
            if re.fullmatch(r'\d{1,3}\.0', n):
                n = n[:-2]
            if re.fullmatch(r'\d[ ,\-]\d[ ,\-]\d', n):
                n = re.sub(r'\D', '', n)
            if not re.fullmatch(r'\d{1,3}', n, flags=re.ASCII):
                raise ValueError('number must contain exactly 1 to 3 decimal digits')
            n = n.zfill(3)
            out.append([d.isoformat(), kind, n, *map(int, n)])
        except ValueError as e:
            rejected.append({'row': str(idx), 'reason': str(e)})
    result = pd.DataFrame(out, columns=COLUMNS)
    duplicate_count = int(result.duplicated(['date', 'draw_type', 'number']).sum())
    result = result.drop_duplicates(['date', 'draw_type', 'number'])
    conflicts = result[result.duplicated(['date', 'draw_type'], keep=False)]
    if len(conflicts):
        raise ValueError('Conflicting results for the same draw: ' + conflicts.to_json(orient='records'))
    result = result.sort_values(['date', 'draw_type']).reset_index(drop=True)
    missing = []
    # Verified schedule applies from 2022-12-12 only. Earlier schedule remains unverified.
    for kind, group in result.groupby('draw_type'):
        start = max(group.date.min(), '2022-12-12')
        present = set(group.date)
        for dt in pd.date_range(start, group.date.max()):
            if kind == 'Day' and (dt.weekday() == 6 or (dt.month == 12 and dt.day == 25)):
                continue
            if str(dt.date()) not in present:
                missing.append({'date': str(dt.date()), 'draw_type': kind})
    quality = {'input_rows': len(frame), 'clean_rows': len(result), 'rejected_rows': rejected,
               'exact_duplicate_drawings_removed': duplicate_count,
               'repeated_winning_numbers': int(result.number.duplicated().sum()),
               'possible_missing_drawings': missing,
               'schedule_note': 'From 2022-12-12: Night daily; Day Monday-Saturday except Christmas. Exceptions may exist. Earlier schedule not verified.',
               'missing_entire_draw_types': sorted({'Day', 'Night'} - set(result.draw_type)),
               'number_note': 'Repeated winning numbers on different draws are valid, not removed.'}
    return result, quality

def source_info(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
