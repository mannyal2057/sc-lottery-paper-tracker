import numpy as np
from sklearn.ensemble import RandomForestClassifier

SEED = 20260909
DIGITS = np.array([[int(c) for c in f'{i:03d}'] for i in range(1000)])
WINDOWS = (25, 50, 100, 250, 500, 1000)

def normalize(p):
    p = np.maximum(np.asarray(p, float), 1e-15)
    p /= p.sum()
    assert p.shape == (1000,) and np.isfinite(p).all() and np.isclose(p.sum(), 1)
    return p

def joint(p):
    return normalize(np.prod(p[np.arange(3)[None, :], DIGITS], axis=1))

def position(history, alpha=1):
    return np.array([(np.bincount(history[:, j], minlength=10) + alpha) /
                     (len(history) + 10 * alpha) for j in range(3)])

def gaps(history):
    return np.array([[len(history)-1-np.flatnonzero(history[:, j] == d)[-1]
                      if np.any(history[:, j] == d) else len(history)
                      for d in range(10)] for j in range(3)])

def features(history):
    """Only preceding draws enter this function. No target or timestamp lookahead."""
    if not len(history):
        raise ValueError('Need at least one previous draw')
    last = history[-1]
    s = np.sort(last)
    patterns = [last.sum(), (last % 2).sum(), (last >= 5).sum(), len(set(last)),
                np.any(np.diff(s) == 1), np.all(np.diff(last) > 0),
                np.all(np.diff(last) < 0), last.max(), last.min(), np.ptp(last),
                last[0]-last[1], last[1]-last[2], last[0]-last[2]]
    return np.concatenate([last, gaps(history).ravel(), patterns,
                           *[position(history[-w:]).ravel() for w in (25, 100, 500)]])

def feature_matrix(history, start=25):
    return np.array([features(history[:t]) for t in range(start, len(history)+1)])

class GapState:
    def __init__(self):
        self.exposures = np.full((3, 3), 100.)
        self.successes = np.full((3, 3), 10.)
        self.last_seen = np.full((3, 10), -1)
        self.n = 0

    def update(self, row):
        t = self.n
        if t >= 25:
            buckets = np.digitize(t-1-self.last_seen, [5, 15])
            for j in range(3):
                self.exposures[j] += np.bincount(buckets[j], minlength=3)
                self.successes[j, buckets[j, row[j]]] += 1
        self.last_seen[np.arange(3), row] = t
        self.n += 1

    def probabilities(self):
        buckets = np.digitize(self.n-1-self.last_seen, [5, 15])
        gp = (self.successes/self.exposures)[np.arange(3)[:, None], buckets]
        return gp/gp.sum(axis=1,keepdims=True)

def basic(history, gap_state=None):
    n = len(history)
    result = {'uniform': np.full(1000, .001), 'historical_position': joint(position(history)),
              'bayes_position_strong': joint(position(history, 10))}
    for w in WINDOWS:
        if n >= w:
            result[f'frequency_{w}'] = joint(position(history[-w:]))
    indices = history @ np.array([100, 10, 1])
    result['bayes_number'] = normalize(np.bincount(indices, minlength=1000) + 1)
    pair = np.ones(1000)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        counts = np.bincount(history[:, a]*10 + history[:, b], minlength=100) + 10
        pair *= counts[DIGITS[:, a]*10 + DIGITS[:, b]]
    result['pairs'] = normalize(np.sqrt(pair))
    transition_pairs = np.ones(1000)
    for a,b in ((0,1),(0,2),(1,2)):
        codes = history[:,a]*10+history[:,b]
        counts = np.bincount(codes[1:][codes[:-1] == codes[-1]],minlength=100)+10
        transition_pairs *= counts[DIGITS[:,a]*10+DIGITS[:,b]]
    result['pair_markov'] = normalize(np.sqrt(transition_pairs))
    cold = 1/position(history[-100:],10)
    result['cold_100'] = joint(cold/cold.sum(axis=1,keepdims=True))
    for order in (1, 2):
        if order == 2 and n < 1000:
            continue
        probs = []
        for j in range(3):
            contexts = np.lib.stride_tricks.sliding_window_view(history[:, j], order+1)
            mask = np.all(contexts[:, :order] == history[-order:, j], axis=1)
            counts = np.bincount(contexts[mask, -1], minlength=10) + 10
            probs.append(counts/counts.sum())
        result[f'markov_{order}'] = joint(np.array(probs))
    cross = np.zeros((3, 10))
    for target in range(3):
        for source in range(3):
            c = np.bincount(history[1:, target][history[:-1, source] == history[-1, source]], minlength=10) + 10
            cross[target] += c/c.sum()/3
    result['cross_markov'] = joint(cross)
    # Gap-based hazard uses observed opportunities, not the gambler's fallacy.
    if gap_state is None:
        gap_state = GapState()
        for row in history:
            gap_state.update(row)
    assert gap_state.n == n
    result['gap_hazard'] = joint(gap_state.probabilities())
    # Probability mass for sum classes is divided by class multiplicity.
    sums = DIGITS.sum(axis=1)
    multiplicity = np.bincount(sums, minlength=28)
    observed = np.bincount(history.sum(axis=1), minlength=28)
    result['sum_pattern'] = normalize((observed[sums]+multiplicity[sums]) / multiplicity[sums])
    return result

class MLModels:
    """Fixed hyperparameters, periodic past-only refits, three position classifiers."""
    def __init__(self, include_xgb=True):
        self.models = {}
        self.last_train_index = -1
        self.include_xgb = include_xgb
        self.unavailable = []

    def fit(self, X, history, t, start=25):
        assert len(X) > t-start and t <= len(history)
        train_x, train_y = X[:t-start], history[start:t]
        assert len(train_x) == len(train_y)
        factories = {'random_forest': lambda: RandomForestClassifier(
            n_estimators=80, max_depth=5, min_samples_leaf=30, random_state=SEED, n_jobs=-1)}
        if self.include_xgb:
            try:
                from xgboost import XGBClassifier
                factories['xgboost'] = lambda: XGBClassifier(n_estimators=60, max_depth=2,
                    learning_rate=.05, reg_lambda=10, subsample=1, colsample_bytree=1,
                    random_state=SEED, n_jobs=2, eval_metric='mlogloss')
            except ImportError:
                self.unavailable = ['xgboost unavailable']
        for name, factory in factories.items():
            if name == 'xgboost' and any(len(np.unique(train_y[:, j])) < 10 for j in range(3)):
                note = 'XGBoost skipped at a refit: training data lacks one or more digit classes; ensemble membership stays frozen.'
                if note not in self.unavailable:
                    self.unavailable.append(note)
                continue
            self.models[name] = [factory().fit(train_x, train_y[:, j]) for j in range(3)]
        self.models['random_forest_rolling500'] = [factories['random_forest']().fit(train_x[-500:],train_y[-500:,j]) for j in range(3)]
        self.last_train_index = t-1

    def predict(self, x, t):
        assert self.last_train_index < t, 'Future training data detected'
        result = {}
        for name, models in self.models.items():
            probs = np.full((3, 10), 1e-6)
            for j, model in enumerate(models):
                probs[j, model.classes_.astype(int)] += model.predict_proba(x[None])[0]
            probs /= probs.sum(axis=1, keepdims=True)
            result[name] = joint(probs)
        return result

    def predict_batch(self, X, first_t):
        assert self.last_train_index < first_t
        result = {}
        for name, models in self.models.items():
            probs = np.full((len(X),3,10),1e-6)
            for j,model in enumerate(models):
                probs[:,j,model.classes_.astype(int)] += model.predict_proba(X)
            probs /= probs.sum(axis=2,keepdims=True)
            result[name] = np.array([joint(p) for p in probs])
        return result
