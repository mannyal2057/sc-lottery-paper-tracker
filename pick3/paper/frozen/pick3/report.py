from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .models import DIGITS
from .backtest import TOP_N, ranking

STYLE = '<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#17253a}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap}.status{background:#fff2cf;padding:20px}</style>'

def html(path, title, body):
    Path(path).write_text('<!doctype html><meta charset="utf-8">'+STYLE+
                         '<h1>'+escape(title)+'</h1>'+body, encoding='utf-8')

def setup(root):
    root = Path(root)
    for p in ('data', 'reports/figures', 'models'):
        (root/p).mkdir(parents=True, exist_ok=True)

def unavailable(root, inventory, reason='No historical SC Pick 3 dataset supplied.'):
    setup(root)
    root = Path(root)
    status = {'status': 'DATA_REQUIRED', 'reason': reason, 'inventory': inventory,
              'drawings': 0, 'best_model': None, 'ensemble': None, 'predictive_signal': None,
              'serial_dependence': None, 'mechanical_bias': None, 'top10': [],
              'note': 'No historical analysis, backtest, or prediction has been performed.'}
    (root/'models/model_results.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    text = ('# SC Pick 3 — data required\n\n'+reason+'\n\n'
        'The supplied attachment is a research specification, not historical results. '
        'No draw data was found in the task workspace. No results were fabricated.\n\n'
        'Strongest model, strongest ensemble, backtest performance, significance, serial dependence, '
        'mechanical bias, and current top-10: **not determinable without data**.\n\n'
        'The pipeline was executed in missing-data mode. Automated tests use explicitly synthetic '
        'fixtures solely to check software behavior; those are not lottery evidence.\n\n'
        'See README.md for input format and execution instructions and IMPLEMENTATION.md for scope.\n\n'
        'For a fair independent drawing, each straight has probability 1/1000. A set of N distinct '
        'straights covers N/1000 of possible outcomes; buying more combinations increases cost. '
        'This framework cannot guarantee a winning number.\n')
    (root/'reports/full_analysis.md').write_text(text, encoding='utf-8')
    (root/'reports/final_prediction.txt').write_text('DATA REQUIRED — No picks generated.\n'+reason, encoding='utf-8')
    for name in ('data_quality_report', 'exploratory_analysis', 'model_comparison'):
        html(root/f'reports/{name}.html', name.replace('_',' ').title(),
             '<div class="status">DATA REQUIRED — No historical results available.</div><pre>'+escape(text)+'</pre>')
    schemas = {'backtest_results': ['draw_type','date','model','phase','actual','prediction','log_loss'],
               'model_scores': ['draw_type','model','test_draws','log_loss','top10_rate'],
               'ensemble_rankings': ['draw_type','number','probability','rank','status']}
    for name, cols in schemas.items():
        pd.DataFrame(columns=cols).to_csv(root/f'reports/{name}.csv', index=False)
    pd.DataFrame(columns=['date','draw_type','number','H','T','U']).to_csv(root/'data/sc_pick3_clean.csv', index=False)
    (root/'reports/figures/README.txt').write_text('Charts require real historical data. None generated in missing-data mode.', encoding='utf-8')
    return status

def charts(root, kind, history, tables, backtest, scores, candidates):
    dest = Path(root)/'reports/figures'
    paths = []
    def save(name):
        path = dest/f'{kind}_{name}.png'
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
        paths.append(path.name)
    plt.figure(figsize=(9,4))
    for j, p in enumerate('HTU'):
        plt.plot(range(10), np.bincount(history[:,j], minlength=10)/len(history), label=p)
    plt.axhline(.1, color='gray', linestyle='--'); plt.legend(); plt.title('Digit frequency by position'); save('digit_frequency')
    plt.figure(figsize=(9,4))
    for d in range(10):
        plt.plot(pd.Series(history[:,0] == d).rolling(100).mean(), label=str(d), alpha=.7)
    plt.legend(ncol=10); plt.title('Hundreds: rolling 100-draw frequency'); save('rolling_frequency')
    plt.figure(figsize=(9,4))
    vals = [int(v) for s in tables['gaps'].completed_gaps for v in s.split(',') if v]
    plt.hist(vals, bins=30); plt.title('Completed digit gaps (draws between appearances)'); save('gap_distribution')
    fig, axes = plt.subplots(3,3,figsize=(10,9))
    for a in range(3):
        for b in range(3):
            c = np.ones((10,10))
            np.add.at(c, (history[:-1,a], history[1:,b]), 1)
            axes[a,b].imshow(c/c.sum(1,keepdims=True), vmin=0, vmax=.2)
            axes[a,b].set_title('HTU'[a]+' → '+'HTU'[b])
    save('transitions')
    fig, axes = plt.subplots(1,3,figsize=(11,3))
    for ax, (name,g) in zip(axes, tables['pairs'].groupby('pair')):
        ax.imshow(g['count'].to_numpy().reshape(10,10)); ax.set_title(name)
    save('pair_frequencies')
    plt.figure(figsize=(9,4)); plt.hist(tables['numbers'].straight_frequency,bins=25)
    plt.title('Historical occurrence counts of 1,000 numbers'); save('number_frequency')
    plt.figure(figsize=(10,6)); plt.barh(scores.model, scores.log_loss); plt.axvline(np.log(1000),color='red')
    plt.xlabel('Test log loss (lower is better)'); save('model_performance')
    test = backtest[(backtest.phase == 'test') & (backtest.model == 'ensemble_static')]
    plt.figure(figsize=(9,4)); plt.plot(np.cumsum(np.log(1000)-test.log_loss.to_numpy()))
    plt.axhline(0,color='gray'); plt.title('Static ensemble cumulative log-score gain versus uniform'); save('walk_forward')
    plt.figure(figsize=(9,4)); row = scores[scores.model == 'ensemble_static'].iloc[0]
    plt.plot(TOP_N,[row[f'top{k}_rate'] for k in TOP_N],label='Static ensemble')
    plt.plot(TOP_N,np.array(TOP_N)/1000,label='Random distinct selections'); plt.legend(); save('top_n')
    plt.figure(figsize=(6,5))
    bins = pd.qcut(test.predicted_top1_probability, q=5, duplicates='drop')
    cal = test.groupby(bins, observed=True).agg(prob=('predicted_top1_probability','mean'),actual=('top1','mean'))
    plt.scatter(cal.prob,cal.actual); plt.plot([0,max(.01,test.predicted_top1_probability.max())],[0,max(.01,test.predicted_top1_probability.max())])
    plt.title('Top-1 reliability (rare hits; descriptive only)'); save('calibration')
    plt.figure(figsize=(9,4)); top = candidates.head(20)
    plt.bar(top.number,top.model_agreement); plt.xticks(rotation=60); plt.title('Top-20 support among base-model top-20s'); save('agreement')
    plt.figure(figsize=(9,4)); plt.hist(candidates.probability,bins=30); plt.title('Experimental ensemble probabilities'); save('candidate_scores')
    return paths

def candidate_table(forecasts):
    p = forecasts['ensemble_static']
    order = ranking(p, 20260909)
    support = np.zeros(1000, int)
    for name, probs in forecasts.items():
        if name != 'uniform' and not name.startswith('ensemble'):
            support[ranking(probs,20260909)[:20]] += 1
    result = pd.DataFrame({'number':[f'{v:03d}' for v in order], 'probability': p[order],
                          'rank':np.arange(1,1001), 'model_agreement': support[order],
                          'status':'EXPERIMENTAL — unconfirmed advantage'})
    result['box'] = [''.join(sorted(v)) for v in result.number]
    boxes = result.groupby('box',as_index=False).agg(probability=('probability','sum'),permutations=('number','size'))
    boxes['fair_probability'] = boxes.permutations/1000
    boxes['box_eligible'] = boxes.permutations > 1
    return result, boxes.sort_values('probability',ascending=False)
