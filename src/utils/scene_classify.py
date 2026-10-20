"""Scene classifiers on the distilled-LSP dataset (scene_dataset).

Folds are grouped by scene seed: one scene appears at every elevation and carrier,
so row-level folds would leak its geometry across train and test.

'hgb' / 'logreg' run on scene rows (scenes.pkl); 'mlp' runs on per-UE link rows (links.pkl),
and scene-level evidence comes only from averaging its posteriors (ue_aggregated_accuracy).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Ray_Tracing.src.utils.scene_classes import CLASSES
from Ray_Tracing.src.utils.scene_dataset import _role, feature_columns

CAT_COLS = ("pol_pair",)
MLP_CAT_COLS = ("pol_pair", "frequency")
ALLOWED_ROLES = ("feature", "context")
LINEAR_PAIRS = ("V/V", "H/H")
CP_PAIRS = ("RHCP/RHCP",)


IMPUTE = {"median": dict(strategy="median"), "mean": dict(strategy="mean"),
          "zero": dict(strategy="constant", fill_value=0.0)}


def make_model(kind="hgb", cols=(), impute="median", add_indicator=True, **mlp_kw):
    """'hgb': gradient boosting (NaN-native); 'logreg': imputed, scaled multinomial logit;
    'mlp': torch MLP (scene_mlp.MLPClassifier) on imputed (+ missing-indicator), scaled inputs.
    impute / add_indicator apply to 'mlp' only; mlp_kw go to MLPClassifier."""
    if kind == "mlp":
        from Ray_Tracing.src.utils.scene_mlp import MLPClassifier
        if impute not in IMPUTE:
            raise ValueError(f"impute must be one of {sorted(IMPUTE)}")
        cat = [c for c in cols if c in MLP_CAT_COLS]
        num = [c for c in cols if c not in MLP_CAT_COLS]
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                                 ("num", make_pipeline(SimpleImputer(add_indicator=add_indicator, **IMPUTE[impute]),
                                                       StandardScaler()), num)], sparse_threshold=0)
        return make_pipeline(pre, MLPClassifier(**mlp_kw))
    cat = [c for c in cols if c in CAT_COLS]
    num = [c for c in cols if c not in CAT_COLS]
    if kind == "hgb":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                                remainder="passthrough", sparse_threshold=0)
        return make_pipeline(pre, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                 random_state=0))
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                             ("num", make_pipeline(SimpleImputer(strategy="median"),
                                                   StandardScaler()), num)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def _check_cols(cols):
    bad = {c: _role(c) for c in cols if _role(c) not in ALLOWED_ROLES}
    assert not bad, f"non-feature columns would reach the model: {bad}"


def _xy(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    return X, df.scene_class.to_numpy()


def _fit(kind, cols, X, y, groups, **mlp_kw):
    m = make_model(kind, cols, **mlp_kw)
    kw = {f"{m.steps[-1][0]}__groups": np.asarray(groups)} if kind == "mlp" else {}
    return m.fit(X, y, **kw)


def _map_folds(fn, items, kind, mlp_kw):
    """[fn(item, **mlp_kw)]; with kind 'mlp' and device='all', items run concurrently, one GPU each."""
    devs = []
    if kind == "mlp" and mlp_kw.get("device") == "all":
        from Ray_Tracing.src.utils.scene_mlp import gpu_devices
        devs = gpu_devices()
    if len(devs) < 2:
        return [fn(it, **mlp_kw) for it in items]

    def worker(j):   # GPU j takes items j, j + n, ...
        return [fn(items[i], **{**mlp_kw, "device": devs[j]}) for i in range(j, len(items), len(devs))]

    out = [None] * len(items)
    with ThreadPoolExecutor(len(devs)) as ex:
        for j, res in enumerate(ex.map(worker, range(len(devs)))):
            for i, r in zip(range(j, len(items), len(devs)), res):
                out[i] = r
    return out


def _with_proba(df, m, X):
    P = m.predict_proba(X)
    return df.assign(pred=m.classes_[P.argmax(1)], **{f"p_{c}": P[:, i] for i, c in enumerate(m.classes_)})


def grouped_cv(df, kind="hgb", cols=None, folds=5, seed=0, shuffle_labels=False, **mlp_kw):
    """Out-of-fold predictions (StratifiedGroupKFold by scene_seed); returns df with 'pred',
    'label' and p_<class> columns."""
    cols = list(cols or feature_columns(df))
    _check_cols(cols)
    X, y = _xy(df, cols)
    if shuffle_labels:   # control: scene-level permutation keeps the group structure
        seeds = df.scene_seed.unique()
        perm = dict(zip(seeds, np.random.default_rng(seed).permutation(
            df.groupby("scene_seed").scene_class.first().loc[seeds].to_numpy())))
        y = df.scene_seed.map(perm).to_numpy()
    g = df.scene_seed.to_numpy()
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)

    def fold(item, **kw):
        k, (tr, te) = item
        assert not set(g[tr]) & set(g[te])
        m = _fit(kind, cols, X.iloc[tr], y[tr], g[tr], **kw)
        return _with_proba(df.iloc[te], m, X.iloc[te]).assign(fold=k, label=y[te], _pos=te)

    parts = _map_folds(fold, list(enumerate(cv.split(X, y, g))), kind, mlp_kw)
    return pd.concat(parts).sort_values("_pos").drop(columns="_pos")


def train_test(train, test, kind="hgb", cols=None, **mlp_kw):
    """Fit on one frame, predict another (cross-nuisance / cross-dataset transfer)."""
    cols = list(cols or feature_columns(train))
    _check_cols(cols)
    m = _fit(kind, cols, *_xy(train, cols), train.scene_seed, **mlp_kw)
    return _with_proba(test, m, _xy(test, cols)[0]).assign(label=test.scene_class.to_numpy())


def accuracy_by(res, by=("elevation", "frequency")):
    return (res.assign(ok=res.pred == res.label).groupby(list(by)).ok
            .agg(accuracy="mean", n="size").reset_index())


def confusion(res, labels=CLASSES):
    return pd.DataFrame(confusion_matrix(res.label, res.pred, labels=list(labels), normalize="true"),
                        index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="pred"))


def overall_accuracy(res):
    return accuracy_score(res.label, res.pred)


def lsp_importance(df, kind="hgb", cols=None, test_frac=0.3, seed=0, n_repeats=5, **mlp_kw):
    """Permutation importance on a held-out scene split, summed per LSP family."""
    cols = list(cols or feature_columns(df))
    _check_cols(cols)
    seeds = df.scene_seed.unique()
    test = np.random.default_rng(seed).choice(seeds, int(len(seeds) * test_frac), replace=False)
    te = df.scene_seed.isin(test)
    m = _fit(kind, cols, *_xy(df[~te], cols), df.scene_seed[~te], **mlp_kw)
    X, y = _xy(df[te], cols)
    imp = permutation_importance(m, X, y, n_repeats=n_repeats, random_state=seed,
                                 n_jobs=1 if kind == "mlp" else -1)   # torch already multithreads
    s = pd.Series(imp.importances_mean, index=cols)
    fam = s.index.str.replace(r"^f_", "", regex=True).str.replace(r"_(all|los|nlos)_.*$", "", regex=True)
    return s.groupby(fam).sum().sort_values(ascending=False)


def curve_importance(df, kind="mlp", cols=None, by="lsp", test_frac=0.3, seed=0, n_repeats=5, **mlp_kw):
    """Grouped permutation importance on elevation-curve rows (scene_dataset.elevation_curves).

    by='lsp': permute one LSP at every elevation together; by='elevation': permute every LSP at
    one elevation together; by='time': every LSP at one walk snapshot ('@t<k>' columns).
    Context columns are their own groups. Returns mean accuracy drop.
    """
    cols = list(cols or feature_columns(df))
    _check_cols(cols)
    i = {"lsp": 0, "elevation": 1, "time": 2}[by]
    groups = {}
    for c in cols:
        p = c.split("@")
        groups.setdefault(p[i] if len(p) > i else c, []).append(c)
    rng = np.random.default_rng(seed)
    seeds = df.scene_seed.unique()
    te = df.scene_seed.isin(rng.choice(seeds, int(len(seeds) * test_frac), replace=False))
    m = _fit(kind, cols, *_xy(df[~te], cols), df.scene_seed[~te], **mlp_kw)
    X, y = _xy(df[te], cols)
    X = X.reset_index(drop=True)
    base = accuracy_score(y, m.predict(X))
    out = {}
    for g, gc in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[gc] = X[gc].to_numpy()[rng.permutation(len(X))]   # same row shuffle for the whole group
            drops.append(base - accuracy_score(y, m.predict(Xp)))
        out[g] = np.mean(drops)
    return pd.Series(out, name=f"importance_by_{by}").sort_values(ascending=False)


def ue_aggregated_accuracy(res, ks=(1, 2, 5, 10, 20), n_draws=20, seed=0, by=("frequency",)):
    """Scene decision from k random UEs of one (scene, elevation, carrier): mean log-posterior,
    argmax. res: per-UE output of grouped_cv / train_test (needs p_<class>)."""
    cls = [c for c in CLASSES if f"p_{c}" in res]   # partial runs may lack a class
    pcols = [f"p_{c}" for c in cls]
    keys = [c for c in ("scene_seed", "elevation", "frequency") if c in res]   # curve rows: no elevation
    lp = np.log(np.clip(res[pcols].to_numpy(), 1e-12, None))
    base = res[keys + ["label"] + [b for b in by if b not in keys]].reset_index(drop=True)
    base[pcols] = lp
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for d in range(n_draws):
            r = base.assign(_u=rng.random(len(base))).sort_values("_u")
            r = r[r.groupby(keys).cumcount() < k]
            g = r.groupby(keys + ["label"] + [b for b in by if b not in keys], sort=False)[pcols].mean()
            pred = np.asarray(cls)[g.to_numpy().argmax(1)]
            rows.append(g.index.to_frame(index=False).assign(ok=pred == g.index.get_level_values("label"),
                                                             k=k, draw=d))
    out = pd.concat(rows, ignore_index=True)
    return out.groupby(["k", *by]).ok.agg(accuracy="mean", n="size").reset_index()


def pol_transfer(cp_links, mixed_links=None, kind="mlp", cols=None, folds=5, seed=0, cp=CP_PAIRS,
                 linear=LINEAR_PAIRS, **mlp_kw):
    """Train on one polarization family, test on held-out scenes of both.

    cp_links: CP rows' source (base: native RHCP/RHCP); mixed_links: linear rows' source
    (base_mixedpol, same scene seeds), or None to take both from cp_links (one mixed-pol sweep,
    disjoint scenes per pair). Folds split the union of scene seeds, so a test scene is unseen
    in either family. Returns per-UE predictions with 'train_pol', 'test_pol', 'fold'.
    """
    mixed_links = cp_links if mixed_links is None else mixed_links
    A = cp_links[cp_links.pol_pair.isin(cp)]
    B = mixed_links[mixed_links.pol_pair.isin(linear)]
    assert len(A) and len(B), "need both CP and linear rows"
    cls_a = A.groupby("scene_seed").scene_class.first()
    cls_b = B.groupby("scene_seed").scene_class.first()
    shared = cls_a.index.intersection(cls_b.index)
    assert (cls_a[shared] == cls_b[shared]).all(), "shared scene seeds carry different labels"
    cols = [c for c in (cols or feature_columns(A)) if c != "pol_pair"]
    _check_cols(cols)
    sets = {"cp": A, "linear": B}
    lab = pd.concat([cls_a, cls_b[~cls_b.index.isin(shared)]])
    seeds = lab.index.to_numpy()
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)

    def job(item, **kw):   # one (fold, training family) fit, tested on both families
        k, src, (tr, te) = item
        s_tr, s_te = set(seeds[tr]), set(seeds[te])
        assert not s_tr & s_te
        train = sets[src]
        m = _fit(kind, cols, *_xy(train[train.scene_seed.isin(s_tr)], cols),
                 train.scene_seed[train.scene_seed.isin(s_tr)], **kw)
        out = []
        for dst, test in sets.items():
            t = test[test.scene_seed.isin(s_te)]
            out.append(_with_proba(t, m, _xy(t, cols)[0]).assign(
                label=t.scene_class.to_numpy(), train_pol=src, test_pol=dst, fold=k))
        return pd.concat(out)

    items = [(k, src, sp) for k, sp in enumerate(cv.split(seeds, lab.to_numpy(), seeds)) for src in sets]
    return pd.concat(_map_folds(job, items, kind, mlp_kw), ignore_index=True)
