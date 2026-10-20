"""Post-mortem of the elevation-curve MLP on base: dataset limit vs classifier limit."""
import sys, time, os, json
sys.path.insert(0, "/path/to")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, r2_score
from Ray_Tracing.src.utils import scene_dataset as sd, scene_classify as sc
from Ray_Tracing.src.utils.scene_classes import CLASSES

D = "/path/to/Ray_Tracing/Results/scene_class"
OUT = f"{D}/analysis/postmortem/base"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

links = pd.read_pickle(f"{D}/dataset/base/links.pkl")
curves = sd.elevation_curves(links)
ELEVS = sorted(links.elevation.unique())
LSP_COLS = [c for c in curves if "@" in c]
COLS = LSP_COLS + ["frequency", "pol_pair"]
NUIS = ["wall_anchor", "roof_anchor", "ground_anchor", "heading_deg", "itu_alpha", "itu_beta", "itu_gamma_fit",
        "wall_scattering_coefficient", "roof_scattering_coefficient", "ground_scattering_coefficient",
        "all_alpha_r", "all_xpd_coefficient"]
meta = links.groupby("scene_seed")[["scene_class", "geo_alpha", "geo_beta", "geo_gamma"] + NUIS].first()
meta["blk"] = np.log(np.sqrt(meta.geo_alpha * meta.geo_beta / 1e6) * meta.geo_gamma)   # log blocking index
MLP_KW = dict(hidden=(128, 128, 64), dropout=0.1, batch_size=256, max_epochs=300, patience=20)
CV = StratifiedGroupKFold(5, shuffle=True, random_state=0)
R = {}


def agg(res, keys=("scene_seed", "frequency")):
    """Scene decision over all UEs: mean log-posterior."""
    pc = [f"p_{c}" for c in CLASSES]
    g = res.assign(**{c: np.log(np.clip(res[c], 1e-12, None)) for c in pc}).groupby(list(keys))
    out = g[pc].mean()
    out["pred"] = np.asarray(CLASSES)[out.to_numpy().argmax(1)]
    out["label"] = g.label.first()
    return out.reset_index()


def cv_run(df, make, cols, groups_fit=False, frac=1.0, train_acc=False):
    """Generic grouped CV (no role check: diagnostics may use labels/nuisances as inputs)."""
    X = df[cols].replace([np.inf, -np.inf], np.nan); y = df.scene_class.to_numpy(); g = df.scene_seed.to_numpy()
    parts, tr_acc = [], []
    for k, (tr, te) in enumerate(CV.split(X, y, g)):
        if frac < 1:
            s = pd.Series(y[tr], index=g[tr]).groupby(level=0).first()
            keep = s.groupby(s).sample(frac=frac, random_state=k).index
            tr = tr[np.isin(g[tr], keep)]
        m = make()
        kw = {f"{m.steps[-1][0]}__groups": g[tr]} if groups_fit else {}
        m.fit(X.iloc[tr], y[tr], **kw)
        P = m.predict_proba(X.iloc[te])
        parts.append(df.iloc[te].assign(pred=m.classes_[P.argmax(1)], label=y[te],
                                        **{f"p_{c}": P[:, i] for i, c in enumerate(m.classes_)}))
        if train_acc:
            tr_acc.append(accuracy_score(y[tr], m.predict(X.iloc[tr])))
    res = pd.concat(parts)
    return res, (np.mean(tr_acc) if train_acc else np.nan)


def summarise(name, res, tr=np.nan, scene_level=False):
    a = sc.overall_accuracy(res)
    row = dict(per_row=a, train=tr)
    if not scene_level:
        ag = agg(res); row["scene_k20"] = sc.overall_accuracy(ag)
        row["suburban_recall_k20"] = (ag[ag.label == "suburban"].pred == "suburban").mean()
    else:
        row["scene_k20"] = a
        row["suburban_recall_k20"] = (res[res.label == "suburban"].pred == "suburban").mean()
    R[name] = row
    log(name, {k: round(v, 3) for k, v in row.items()})
    pd.DataFrame(R).T.to_csv(f"{OUT}/A_B_models.csv")


# ---------------- A: classifier sweep on the same per-UE curve inputs ----------------
mlp = lambda **kw: (lambda: sc.make_model("mlp", COLS, **{**MLP_KW, **kw}))
res_mlp, tr = cv_run(curves, mlp(), COLS, groups_fit=True, train_acc=True)
res_mlp.to_pickle(f"{OUT}/mlp_default_cv.pkl")
summarise("A mlp default (128,128,64)", res_mlp, tr)
for name, kw in {"A mlp small (64,)": dict(hidden=(64,)),
                 "A mlp big (512,512,256,128) do0.2": dict(hidden=(512, 512, 256, 128), dropout=0.2),
                 "A mlp heavy-reg do0.4 wd1e-2": dict(dropout=0.4, weight_decay=1e-2),
                 "A mlp no-reg do0 wd0": dict(dropout=0.0, weight_decay=0.0)}.items():
    r, tr = cv_run(curves, mlp(**kw), COLS, groups_fit=True, train_acc=True)
    summarise(name, r, tr)
r, tr = cv_run(curves, lambda: sc.make_model("logreg", COLS), COLS, train_acc=True)
summarise("A logreg (linear)", r, tr)
r, tr = cv_run(curves, lambda: sc.make_model("hgb", COLS), COLS, train_acc=True)
summarise("A hgb", r, tr)

# ---------------- B: scene-level information bounds ----------------
scene_mean = curves.groupby(["scene_seed", "frequency"])[LSP_COLS].mean().reset_index().merge(
    meta.reset_index(), on="scene_seed")
scene_mean["pol_pair"] = "RHCP/RHCP"
r, tr = cv_run(scene_mean, lambda: sc.make_model("hgb", LSP_COLS + ["frequency"]), LSP_COLS + ["frequency"], train_acc=True)
summarise("B hgb on scene-mean curves (20 UEs averaged first)", r, tr, scene_level=True)
r, tr = cv_run(scene_mean, lambda: sc.make_model("mlp", LSP_COLS + ["frequency"], **{**MLP_KW, "batch_size": 64}),
               LSP_COLS + ["frequency"], groups_fit=True, train_acc=True)
summarise("B mlp on scene-mean curves", r, tr, scene_level=True)

S = meta.reset_index()
hgb = lambda: __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline(
    HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0))
for name, cols in {"B oracle: drawn geometry (alpha,beta,gamma)": ["geo_alpha", "geo_beta", "geo_gamma"],
                   "B oracle: blocking index only": ["blk"],
                   "B oracle: realised P.1410 (itu_*)": ["itu_alpha", "itu_beta", "itu_gamma_fit"]}.items():
    r, tr = cv_run(S, hgb, cols, train_acc=True); summarise(name, r, tr, scene_level=True)
Sm = pd.get_dummies(S, columns=["wall_anchor", "roof_anchor", "ground_anchor"], dtype=float)
mat_cols = [c for c in Sm if c.startswith(("wall_anchor_", "roof_anchor_", "ground_anchor_"))]
r, tr = cv_run(Sm, hgb, mat_cols, train_acc=True); summarise("B oracle: material anchors only", r, tr, scene_level=True)

# ---------------- C: how much geometry the LSP curves retain ----------------
rows = []
for tgt in ("blk", "geo_alpha", "geo_beta", "geo_gamma"):
    y = np.log(scene_mean[tgt]) if tgt != "blk" else scene_mean[tgt]
    pred = np.empty(len(y))
    for tr_, te_ in CV.split(scene_mean, scene_mean.scene_class, scene_mean.scene_seed):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        m.fit(scene_mean[LSP_COLS + ["frequency"]].iloc[tr_], y.iloc[tr_])
        pred[te_] = m.predict(scene_mean[LSP_COLS + ["frequency"]].iloc[te_])
    rows.append(dict(target=f"log {tgt}" if tgt != "blk" else "log blocking index", R2=r2_score(y, pred),
                     resid_sd=np.std(y - pred), total_sd=np.std(y),
                     within_class_sd=scene_mean.assign(_y=y).groupby("scene_class")._y.std().mean()))
    if tgt == "blk":
        scene_mean["blk_hat"] = pred
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/C_geometry_recovery.csv", index=False); log("C geometry from LSPs\n", C.round(3))
# class from the recovered blocking index alone: does a 1-D geometry estimate reach the MLP's accuracy?
r, _ = cv_run(scene_mean, hgb, ["blk_hat"]); summarise("C hgb on LSP-estimated blocking index only", r, scene_level=True)
# class overlap of the true blocking index vs the estimated one
ov = scene_mean.groupby("scene_class")[["blk", "blk_hat"]].describe(percentiles=[.1, .5, .9]).T
ov.to_csv(f"{OUT}/C_blocking_index_by_class.csv"); log("C blocking index by class\n", ov.round(3))

# ---------------- D: per-UE noise vs class separation ----------------
rows = []
for c in LSP_COLS:
    v = curves[["scene_seed", "frequency", "scene_class", c]].dropna()
    if v[c].nunique() < 3:
        continue
    for f, vf in v.groupby("frequency"):
        tot = vf[c].var()
        within_scene = vf.groupby("scene_seed")[c].var().mean()
        smean = vf.groupby("scene_seed").agg(m=(c, "mean"), cls=("scene_class", "first"))
        between_class = smean.groupby("cls").m.mean().var()
        within_class_scene = smean.groupby("cls").m.var().mean()
        rows.append(dict(col=c, lsp=c.split("@")[0], elev=float(c.split("@")[1]), freq_GHz=f / 1e9,
                         frac_within_scene=within_scene / tot,
                         fisher_per_ue=between_class / (within_scene + within_class_scene),
                         fisher_scene_mean=between_class / within_class_scene))
Dn = pd.DataFrame(rows); Dn.to_csv(f"{OUT}/D_noise_separation.csv", index=False)
log("D median share of variance that is within-scene (UE scatter), per LSP\n",
    Dn.groupby("lsp").frac_within_scene.median().round(3).sort_values())
log("D best Fisher ratios (between-class / within-class), per UE vs scene mean\n",
    Dn.sort_values("fisher_scene_mean", ascending=False).head(12)[["col", "freq_GHz", "fisher_per_ue", "fisher_scene_mean"]].round(3))

# ---------------- E: which scenes fail ----------------
ag = agg(res_mlp).merge(meta.reset_index(), on="scene_seed")
ag["err"] = ag.pred != ag.label
ag.to_csv(f"{OUT}/E_scene_predictions.csv", index=False)
log("E scene-level confusion (all 20 UEs)\n", sc.confusion(ag).round(3))
ag["blk_q"] = ag.groupby("scene_class").blk.transform(lambda s: pd.qcut(s, 5, labels=False))
E1 = ag.groupby(["scene_class", "blk_q"]).agg(err=("err", "mean"), pred_rural=("pred", lambda p: (p == "rural").mean()),
                                              pred_urban=("pred", lambda p: (p == "urban").mean()), blk=("blk", "mean"))
E1.to_csv(f"{OUT}/E_error_vs_blocking_quintile.csv"); log("E error vs within-class blocking-index quintile\n", E1.round(3))
for role in ("wall_anchor", "roof_anchor", "ground_anchor"):
    t = ag.groupby(["scene_class", role]).agg(err=("err", "mean"), n=("err", "size"))
    t.to_csv(f"{OUT}/E_error_by_{role}.csv"); log(f"E error by {role}\n", t.round(3))
# do nuisances predict errors beyond geometry? grouped-CV AUC of an error model
from sklearn.metrics import roc_auc_score
Xe = pd.get_dummies(ag[["scene_class", "blk", "geo_alpha", "geo_beta", "geo_gamma", "frequency", "heading_deg",
                        "wall_anchor", "roof_anchor", "ground_anchor", "wall_scattering_coefficient",
                        "roof_scattering_coefficient", "ground_scattering_coefficient", "all_alpha_r",
                        "all_xpd_coefficient"]], dtype=float)
sets = {"geometry": ["blk", "geo_alpha", "geo_beta", "geo_gamma"],
        "materials": [c for c in Xe if "anchor" in c or "scattering" in c or c in ("all_alpha_r", "all_xpd_coefficient")],
        "all": list(Xe.columns)}
erows = []
for name, cols in sets.items():
    cols = cols + [c for c in Xe if c.startswith("scene_class_")] if name != "all" else cols
    p = np.empty(len(ag))
    for tr_, te_ in CV.split(Xe, ag.scene_class, ag.scene_seed):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0).fit(Xe[cols].iloc[tr_], ag.err.iloc[tr_])
        p[te_] = m.predict_proba(Xe[cols].iloc[te_])[:, 1]
    erows.append(dict(predictors=name, auc=roc_auc_score(ag.err, p)))
E2 = pd.DataFrame(erows); E2.to_csv(f"{OUT}/E_error_predictability.csv", index=False); log("E which draws predict an error (AUC)\n", E2.round(3))

# ---------------- F: learning curve ----------------
lc = []
for frac in (0.125, 0.25, 0.5, 0.75, 1.0):
    r, _ = cv_run(curves, mlp(), COLS, groups_fit=True, frac=frac) if frac < 1 else (res_mlp, None)
    lc.append(dict(train_frac=frac, train_scenes=int(frac * 0.8 * curves.scene_seed.nunique()),
                   per_ue=sc.overall_accuracy(r), scene_k20=sc.overall_accuracy(agg(r))))
    log("F", lc[-1])
pd.DataFrame(lc).to_csv(f"{OUT}/F_learning_curve.csv", index=False)
log("done")
