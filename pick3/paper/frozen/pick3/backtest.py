from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import binomtest
from .models import DIGITS, SEED, basic, feature_matrix, MLModels, normalize, GapState

TOP_N = (1, 5, 10, 20, 50, 100)

def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted

def ranking(prob, seed):
    # Random seeded ties avoid a spurious systematic preference for 000.
    return np.lexsort((np.random.default_rng(seed).random(1000), -prob))

def evaluate(prob, actual, seed):
    rank = ranking(prob, seed)
    pred = rank[0]
    overlap = sum((Counter(DIGITS[pred]) & Counter(DIGITS[actual])).values())
    result = {'prediction': f'{pred:03d}', 'actual': f'{actual:03d}',
              'log_loss': -np.log(prob[actual]),
              'brier': float(np.sum(prob**2)-2*prob[actual]+1),
              'box_hit': int(sorted(DIGITS[pred]) == sorted(DIGITS[actual])),
              'box_chance': {1: .001, 2: .003, 3: .006}[len(set(DIGITS[pred]))],
              'digits_matched': overlap, 'rank': int(np.flatnonzero(rank == actual)[0])+1,
              'predicted_top1_probability': prob[pred]}
    result.update({f'top{k}': int(actual in rank[:k]) for k in TOP_N})
    for a,b,label in ((0,1,'HT'),(0,2,'HU'),(1,2,'TU')):
        masses = np.bincount(DIGITS[:,a]*10+DIGITS[:,b],weights=prob,minlength=100)
        actual_pair = DIGITS[actual,a]*10+DIGITS[actual,b]
        result[f'pair_{label}_log_loss'] = -np.log(masses[actual_pair])
    return result

def weights(actual_probs):
    if len(actual_probs) == 0:
        raise ValueError('Ensemble requires preceding validation predictions')
    m = actual_probs.shape[1]
    fit = minimize(lambda w: -np.log(np.maximum(actual_probs @ w, 1e-15)).mean(),
                   np.full(m, 1/m), method='SLSQP', bounds=[(0, 1)]*m,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1}],
                   options={'maxiter': 250, 'ftol': 1e-10})
    if not fit.success:
        raise RuntimeError('Ensemble optimization failed: ' + fit.message)
    return np.maximum(fit.x, 0)/np.maximum(fit.x, 0).sum()

def block_interval(values, seed=SEED, draws=1000):
    """Circular moving-block bootstrap, approximately sqrt(n) draw blocks."""
    values = np.asarray(values)
    n = len(values)
    rng = np.random.default_rng(seed)
    width = max(1, int(np.sqrt(n)))
    means = []
    for _ in range(draws):
        starts = rng.integers(n, size=int(np.ceil(n/width)))
        ix = (starts[:, None] + np.arange(width)) % n
        means.append(values[ix.ravel()[:n]].mean())
    return np.quantile(means, [.025, .975]).tolist()

def run(history, dates, initial=1000, validation=250, refit=100, ml=True):
    if len(history) < initial + validation + 100:
        raise ValueError(f'Need at least {initial+validation+100} drawings per draw type')
    assert initial >= 25 and validation >= 20 and refit >= 1
    cutoff = initial+validation
    X = feature_matrix(history) if ml else None
    engine = MLModels()
    actual_probs, records = [], []
    model_names, static = None, None
    last_probabilities = None
    state = GapState()
    for row in history[:initial]:
        state.update(row)
    for t in range(initial, len(history)+1):
        forecasts = basic(history[:t], state)
        if ml:
            if t == initial or (t-initial) % refit == 0 or t == len(history):
                engine.fit(X, history, t)
                block_start = t
                # Batch inference only; each row's features still use its own past.
                batch = engine.predict_batch(X[t-25:min(t+refit,len(history)+1)-25],t)
            forecasts.update({name: p[t-block_start] for name,p in batch.items()})
        if (t-initial) % 500 == 0:
            print(f'  Forecast {t}/{len(history)}',flush=True)
        if model_names is None:
            model_names = sorted(forecasts)
        # Freeze candidate membership before validation; no later test-driven additions.
        forecasts = {name: forecasts[name] for name in model_names}
        mat = np.array([forecasts[name] for name in model_names])
        if t >= cutoff:
            if static is None:
                static = weights(np.array(actual_probs))
            forecasts['ensemble_static'] = normalize(static @ mat)
            for w in (100, 250, 500):
                past = np.array(actual_probs[-w:])
                dynamic = softmax(np.log(np.maximum(past, 1e-15)).sum(axis=0))
                forecasts[f'ensemble_dynamic_{w}'] = normalize(dynamic @ mat)
        if t == len(history):
            last_probabilities = forecasts
            break
        y = int(history[t] @ np.array([100, 10, 1]))
        for name, p in forecasts.items():
            records.append({'draw_index': t, 'date': dates[t], 'model': name,
                            'phase': 'validation' if t < cutoff else 'test',
                            'trained_through': dates[t-1], **evaluate(p, y, SEED+t)})
        # The label is consumed only AFTER all forecasts and ensemble weights are fixed.
        actual_probs.append(mat[:, y])
        state.update(history[t])
    return pd.DataFrame(records), last_probabilities, dict(zip(model_names, static.tolist())), engine.unavailable

def summarize(frame):
    test = frame[frame.phase == 'test']
    summary = []
    for name, group in test.groupby('model'):
        row = {'model': name, 'test_draws': len(group),
               'log_loss': group.log_loss.mean(), 'brier': group.brier.mean(),
               'box_hit_rate': group.box_hit.mean(), 'box_chance': group.box_chance.mean(),
               'average_digits_matched': group.digits_matched.mean(),
               'mean_rank': group['rank'].mean()}
        for pair in ('HT','HU','TU'):
            row[f'pair_{pair}_log_loss'] = group[f'pair_{pair}_log_loss'].mean()
        for d in range(4):
            row[f'overlap_{d}_rate'] = (group.digits_matched == d).mean()
        gain = np.log(1000)-group.log_loss.to_numpy()
        row['log_gain_ci_low'], row['log_gain_ci_high'] = block_interval(gain)
        for k in TOP_N:
            hits = int(group[f'top{k}'].sum())
            bt = binomtest(hits, len(group), k/1000, alternative='greater')
            ci = binomtest(hits, len(group)).proportion_ci()
            row.update({f'top{k}_rate': hits/len(group), f'top{k}_chance': k/1000,
                        f'top{k}_p': bt.pvalue, f'top{k}_ci_low': ci.low, f'top{k}_ci_high': ci.high})
        for i, period_indices in enumerate(np.array_split(np.arange(len(group)), 3)):
            row[f'period_{i+1}_log_gain'] = np.log(1000)-group.iloc[period_indices].log_loss.mean()
        summary.append(row)
    scores = pd.DataFrame(summary)
    # Family includes every model and every top-N selection tested.
    fields = [f'top{k}_p' for k in TOP_N]
    adjusted = holm(scores[fields].to_numpy().ravel()).reshape(len(scores), len(fields))
    for j, k in enumerate(TOP_N):
        scores[f'top{k}_p_holm'] = adjusted[:, j]
    return scores.sort_values('log_loss')
