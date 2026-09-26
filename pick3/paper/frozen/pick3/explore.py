import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency, binomtest
from .models import DIGITS, WINDOWS, gaps
from .backtest import holm

def analyze(history):
    n = len(history)
    frequency, gap_rows, tests, pair_rows = [], [], [], []
    for w in [n]+[w for w in WINDOWS if w < n]:
        for j, name in enumerate('HTU'):
            counts = np.bincount(history[-w:, j], minlength=10)
            for d, count in enumerate(counts):
                ci = binomtest(int(count), w, .1).proportion_ci()
                frequency.append([w, name, d, count, count/w, count/w-.1,
                                  ci.low, ci.high, (count-w*.1)/np.sqrt(w*.09)])
            tests.append([f'digit_uniform_{name}_{w}', float(chisquare(counts).pvalue)])
    for j, name in enumerate('HTU'):
        for d in range(10):
            hits = np.flatnonzero(history[:, j] == d)
            intervals = np.diff(hits)-1
            gap_rows.append([name, d, int(gaps(history)[j, d]),
                             intervals.mean() if len(intervals) else None,
                             np.median(intervals) if len(intervals) else None,
                             intervals.min() if len(intervals) else None,
                             intervals.max() if len(intervals) else None,
                             ','.join(map(str, intervals))])
        for k, target in enumerate('HTU'):
            c = np.zeros((10, 10))
            np.add.at(c, (history[:-1, j], history[1:, k]), 1)
            if (c.sum(0)>0).all() and (c.sum(1)>0).all():
                _, p, _, expected = chi2_contingency(c)
                if expected.min() >= 5:
                    tests.append([f'transition_{name}_to_{target}', float(p)])
    indices = history @ np.array([100, 10, 1])
    all_gaps = gaps(history)
    numbers = []
    for v, digits in enumerate(DIGITS):
        at = np.flatnonzero(indices == v)
        interval = np.diff(at)-1
        box = np.all(np.sort(history, axis=1) == np.sort(digits), axis=1)
        numbers.append({'number': f'{v:03d}', 'H': digits[0], 'T': digits[1], 'U': digits[2],
            'straight_frequency': len(at), 'recent_100_frequency': int((indices[-100:] == v).sum()),
            'last_draw_index': int(at[-1]) if len(at) else None,
            'current_gap': n-1-int(at[-1]) if len(at) else None,
            'never_seen': len(at) == 0, 'mean_gap': interval.mean() if len(interval) else None,
            'box_frequency': int(box.sum()), 'sum': int(digits.sum()), 'odd': int((digits%2).sum()),
            'high': int((digits>=5).sum()), 'unique': len(set(digits)),
            'double': len(set(digits)) == 2, 'triple': len(set(digits)) == 1,
            'consecutive': bool(np.any(np.diff(np.sort(digits)) == 1)),
            'ascending': bool(np.all(np.diff(digits)>0)), 'descending': bool(np.all(np.diff(digits)<0)),
            'minimum': int(digits.min()), 'maximum': int(digits.max()), 'range': int(np.ptp(digits)),
            'HT_difference': int(digits[0]-digits[1]), 'TU_difference': int(digits[1]-digits[2]),
            'HU_difference': int(digits[0]-digits[2])})
    for a, b in ((0, 1), (0, 2), (1, 2)):
        codes = history[:, a]*10+history[:, b]
        for v in range(100):
            at = np.flatnonzero(codes == v)
            pair_rows.append(['HTU'[a]+'HTU'[b], f'{v:02d}', len(at), n/100,
                              len(at)-n/100, n-1-int(at[-1]) if len(at) else None])
    testframe = pd.DataFrame(tests, columns=['test', 'p_value'])
    testframe['p_holm'] = holm(testframe.p_value)
    return {
        'digit_frequencies': pd.DataFrame(frequency, columns=['window', 'position', 'digit', 'count', 'fraction', 'deviation', 'ci_low', 'ci_high', 'z_score']),
        'gaps': pd.DataFrame(gap_rows, columns=['position','digit','current_gap','mean_gap','median_gap','min_gap','max_gap','completed_gaps']),
        'randomness_tests': testframe,
        'numbers': pd.DataFrame(numbers),
        'pairs': pd.DataFrame(pair_rows, columns=['pair', 'digits', 'count', 'expected', 'deviation', 'current_gap'])}
