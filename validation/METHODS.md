# RT vs TR 38.811: definitions, estimators, and what each comparison is worth

Companion to `validate_3gpp.py`, `lsp_stats.py`, `cluster_stats.py` and
`cluster_id.py`. Every number quoted here is measured on `results_sieg_2GHz`
(UW campus, 10 000 receivers per elevation, 500 km slant range) unless stated.

The recurring theme: TR 38.811 tabulates parameters of a *generative* model.
Ray tracing produces one realisation of one scene. Several of the comparisons in
this repo pair a spec parameter with an RT estimator of a different quantity, and
the useful output of a validation run is knowing which ones those are.

---

## 1. Shadow fading and clutter loss

### 1.1 Definition

TR 38.811 §6.6.1 decomposes the link budget as `PL = PL_b + PL_g + PL_s + PL_e`
(basic, gaseous, scintillation, building entry), with

```
PL_b(d, alpha) = FSPL(d, f_c) + SF + CL(alpha, f_c)
FSPL(d, f_c)   = 32.45 + 20 log10 f_c[GHz] + 20 log10 d[m]
SF ~ N(0, sigma_SF^2)          CL(alpha, f_c) tabulated, CL == 0 for LOS
```

so the observable is conditionally Gaussian:

```
PL_b | LOS  ~ N( FSPL,               sigma_SF,LOS^2 )
PL_b | NLOS ~ N( FSPL + CL(alpha),   sigma_SF,NLOS^2 )
```

**`sigma_SF` and `CL` are the variance and the mean of the same conditional
distribution.** Table 6.6.2-1/-2 tabulates them together per (elevation, band,
state); ns-3 carries the pair as `SFCL_SuburbanRural`, which
`ns3_reference.parse_sfcl` parses into `sigma_SF_dB` and `clutter_loss_dB`.

### 1.2 Estimators

RT gives `L_i = -10 log10(P_los,i + P_nlos,i)` per receiver, an incoherent path-power
sum. `lsp_stats.per_rx_lsp_frame` forms the zero-mean residual within each state:

```
SF_hat_i    = L_i - mean_{j in S} L_j
sigma_SF^2  = (1/|S|) sum_{i in S} SF_hat_i^2
```

Over a 340 m scene at 500 km the geometric range spread is
`20 log10(1 + 7e-4) ~ 6e-3 dB`, so FSPL is common to every receiver and the two
state means are `FSPL` and `FSPL + E[CL]`. That gives the clutter-loss estimator,
which until now was never formed:

```
CL_hat(alpha) = mean_loss_db(NLOS) - mean_loss_db(LOS)
```

Both terms already existed in the LSP table. `validate_3gpp.compare_clutter_loss`
now scores it, writing `clutter_loss.csv`.

### 1.3 Measured

| elev | RT CL_hat | ref CL | delta | RT sigma_SF LOS | ref | RT sigma_SF NLOS | ref |
|---|---|---|---|---|---|---|---|
| 10 | 23.89 | 19.52 | +4.37 | 0.74 | 1.79 | 6.61 | 8.93 |
| 20 | 22.69 | 18.17 | +4.52 | 0.31 | 1.14 | 5.38 | 9.08 |
| 30 | 21.36 | 18.42 | +2.94 | 0.18 | 1.14 | 5.73 | 8.78 |
| 40 | 20.42 | 18.28 | +2.14 | 0.14 | 0.92 | 5.59 | 10.25 |
| 50 | 20.05 | 18.63 | +1.42 | 0.13 | 1.42 | 5.34 | 10.56 |
| 60 | 19.69 | 17.68 | +2.01 | 0.15 | 1.56 | 5.93 | 10.74 |
| 70 | 19.80 | 16.50 | +3.30 | 0.17 | 0.85 | 6.79 | 10.17 |
| 80 | 20.69 | 16.30 | +4.39 | 0.17 | 0.72 | 10.79 | 11.52 |
| 90 | — | — | — | 0.03 | 0.72 | — | — |

90 deg is withheld: 48 NLOS receivers, below `validate_3gpp.MIN_N = 200`.

Clutter loss agrees to 1.4-4.5 dB with the right elevation trend (monotone
decrease to ~60 deg). `sigma_SF` does not, and the reason is structural, not a bug.

### 1.4 Why sigma_SF is not a fair comparison

Index environment realisations by `E` and UE positions by `u`. The tabulated
quantity is a total variance; by the law of total variance,

```
sigma_3GPP^2 = Var_{E,u}[L] = E_E[ Var_u[L | E] ]  +  Var_E[ E_u[L | E] ]
                              \___ within-scene ___/   \__ between-scene __/
```

One fixed campus estimates only the first term, so `sigma_SF_hat <= sigma_3GPP`
**by construction**. Measured LOS 0.03-0.74 dB against a tabulated 0.72-1.79 dB is
the between-scene term made visible. `validate_3gpp` therefore reports SF with
`status = "scene-limited"` and never scores it. The same ceiling bounds the SF
decorrelation distance: 11.5 m measured against a tabulated 37 m, over a 340 m scene.

A second, smaller effect: 3GPP removes small-scale fading by local averaging over
10-40 wavelengths; the RT residual does not. The local-median residual
`SF_dB_smooth` (5 m radius) isolates the sub-5 m share, so

```
sigma_shadow = sqrt( sigma_SF^2 - sigma_SF_smooth^2 )
```

At 30 deg NLOS: `sqrt(5.73^2 - 2.96^2) = 4.90 dB`. `sigma_SF_dB_smooth` is carried
in the `rt_alt` column of the SF rows so both are visible together.

---

## 2. The LSP cross-correlation matrix

### 2.1 Definition

3GPP models the LSP vector as jointly Gaussian and generates it by a Cholesky
factor:

```
s = mu + diag(sigma) L z,     z ~ N(0, I),     C = L L^T
s = [SF, K, lgDS, lgASD, lgASA, lgZSD, lgZSA]      (LOS, 7x7)
s = [SF,    lgDS, lgASD, lgASA, lgZSD, lgZSA]      (NLOS, 6x6, no K)
```

`L` is ns-3's `sqrtC_NTN_Suburban_{LOS,NLOS}`; `ns3_reference.correlation_from_sqrt`
recovers `C`. RT estimates `C` as the sample correlation over receivers.

**ns-3 stores one elevation-independent matrix for Suburban**, so every elevation's
RT matrix is scored against the same reference, even though TR 38.811 Table
6.7.2-5/-6 tabulates per elevation. Variation of `frobenius` with elevation is
entirely an RT-side effect. (ns-3 does keep per-elevation maps for Urban/Rural
NLOS; `ns3_reference.correlation_frame` cannot parse those yet.)

### 2.2 A selection-bias defect, now fixed

The all-finite row mask used to be computed over all seven columns, *including*
the degenerate `lgASD`/`lgZSD`. Since `ASD_i = 0 => lgASD_i = -inf`, this
conditioned the estimate on the event

```
A = { ASD_i > 0 } and { ZSD_i > 0 }
```

so the estimand was `Corr(s_a, s_b | A)`, not `Corr(s_a, s_b)`. For a jointly
Gaussian `s`, conditioning on any event correlated with its components shifts both
the mean and the covariance — and `A` is strongly informative, because a nonzero
departure spread needs two resolvable exit directions at the satellite, i.e. more
scattering, which correlates with DS, ASA and K.

Measured retention at 2 GHz / 30 deg: **2680 of 8082 LOS receivers (33%)** and
948 of 1918 NLOS (49%). `lsp_stats.lsp_correlation` now chooses the column set
first and masks rows only over the surviving columns, restoring 8081 and 1914.
The effective sample size is reported as `n_rx` in `correlation_*.csv`.

Effect on the scored quantity (Frobenius norm of the label-aligned delta,
ASD/ZSD excluded):

| elev | LOS before | LOS after | NLOS before | NLOS after |
|---|---|---|---|---|
| 10 | 2.315 | 2.198 | 1.604 | 1.730 |
| 30 | 2.244 | 2.262 | 1.274 | 1.321 |
| 60 | 2.582 | 2.561 | 1.165 | 1.196 |
| 90 | 2.726 | 2.584 | 1.521 | 1.413 |

Honest reading: the conclusion does not change — RT and TR 38.811 disagree at
mean |delta| ~ 0.3-0.4 either way. What changes is that the estimate is now of the
quantity it claims to be, computed from three times the data.

### 2.3 Degeneracy is decided once, over both states

The rule is `frac(raw <= 0) > 0.5`. Measured per-state fractions straddle the
threshold — ASD is 0.668 non-positive in LOS but 0.491 in NLOS at 2 GHz / 30 deg —
so the same physical quantity was classed degenerate in one state and live in the
other within a single table. `lsp_stats.degenerate_spreads` is now evaluated once
per (config, elevation) over the pooled receiver set and passed to both
`lsp_marginals` and `lsp_correlation`.

---

## 3. Decorrelation distance

`rho(d) = exp(-d / d_cor)` fitted to binned standardised products. Two structural
negative biases, both inherent to the estimator:

1. **Sample-mean removal.** With `sum z_i = 0` and `sum z_i^2 = N`,
   `sum_{i<j} z_i z_j = ((sum z_i)^2 - sum z_i^2)/2 = -N/2`, so the all-pairs mean
   of `rho_hat` is exactly `-1/(N-1)`. The estimator is pushed negative at large
   lag regardless of the true field.
2. **Bounded domain.** Bins approaching the scene diameter hold few, edge-biased
   pairs.

Measured net bias on synthetic fields with a planted `d_cor = 30 m`: -17.8% fitting
the full 150 m window, -8.3% with the adaptive `rho_floor` truncation, with ~20%
single-realisation scatter. Quote `d_cor` with that uncertainty and always show the
raw `rho(d)` curve, which over one fixed building layout is frequently not a clean
exponential.

---

## 4. What a "cluster" is in TR 38.811

§6.7.2 inherits TR 38.901 §7.5 with NTN tables. A cluster is a **latent variable in
a generative model**; the spec defines no measurement operator.

**Step 5 — delays.** `tau'_n = -rTau * DS * ln(X_n)`, `X_n ~ U(0,1)`, then
`tau_n = sort(tau') - min(tau')`. So `tau'_n ~ Exp(m)` with `m = rTau * DS`.

**Step 6 — powers.**
`P'_n = exp(-tau_n (rTau-1)/(rTau*DS)) * 10^(-Z_n/10)`, `Z_n ~ N(0, zeta^2)`,
`zeta = perClusterShadowingStd = 3 dB`; normalise, then **discard clusters with
`P_n < 10^-2.5 * max P`**. LOS folds in the specular ray scaled by `K_R/(K_R+1)`.

**Step 7 — angles.** Cluster angles are drawn from the total ASA/ZSA by an inverse
Gaussian/Laplacian map, then `M = 20` rays per cluster at fixed offsets
`+/- cASA * alpha_m`, `+/- cZSA * alpha_m` (Table 7.5-3).

Consequences for validation:

* `numOfCluster` is a model parameter, not an observable. Any RT number depends on
  an identification procedure the spec does not supply, so the procedure has to be
  argued for on its own terms.
* The tabulated `cASA = 11 deg` / `cZSA = 7 deg` are TR 38.901 terrestrial
  constants and **exceed the tabulated total ASA/ZSA at every elevation in both
  bands** (LOS/S at 30 deg: total ASA `10^-0.56 = 0.28 deg` against `cASA = 11 deg`).
  An intra-cluster spread cannot exceed the total. `validate_3gpp` marks these
  `"inconsistent reference"`; they are not usable targets regardless of clustering.
* `cASD = 0` is vacuous under NOTE 8; `raysPerCluster = 20` is not an observable.

That leaves `numOfCluster`, `rTau`, `cDS` and `perClusterShadowingStd` as the four
scoreable cluster parameters.

---

## 5. Cluster identification

### 5.1 The baseline and why it fails

`cluster_stats.py` bins paths into a 1 ns PDP, finds peaks at 3 dB prominence, and
assigns each path to the nearest peak in delay. Delay-only; angles are summarised
afterwards but never used to form clusters. RT diffuse scattering fills delay
continuously (~99% of stored paths are diffuse), so the peak count is a property of
`prominence_db`, not of the channel: it returns 6.6 (LOS) / 11.7 (NLOS) against a
tabulated 3 / 4.

### 5.2 MCD

`cluster_id.py` clusters jointly in delay and arrival direction using the
Multipath Component Distance. Departure angles are dropped — measured spread within
one receiver is 0.0006 deg in zenith and 0.002 deg in azimuth, i.e. NOTE 8 as data.

```
Omega_i        = (sin th_i cos ph_i, sin th_i sin ph_i, cos th_i)
MCD_Omega_ij   = ||Omega_i - Omega_j|| / 2 = sin(dpsi_ij / 2)          in [0,1]
MCD_tau_ij     = zeta * (sigma_Omega / sigma_tau) * |tau_i - tau_j|
MCD_ij         = sqrt( MCD_Omega_ij^2 + MCD_tau_ij^2 )
```

with `sigma_Omega`, `sigma_tau` the power-weighted spreads. The chord form needs no
wrap convention, which is why it is correct for zenith where the circular estimator
`sqrt(-2 ln |E[e^{j th}]|)` — applied to zenith by TR 38.901 and inherited by
`channel_utils.rms_angular_spread` — is not.

**Delay normalisation.** Czink's published `sigma_tau / span^2` form is selectable
but wrong for this data: `span ~ 470 ns` against `sigma_tau ~ 28 ns` compresses the
delay axis by `(span/sigma_tau)^2 ~ 280` and the clustering degenerates to
angle-only. Whitening by `sigma_tau` is robust to the heavy diffuse tail.
`zeta = 1` is the neutral choice — one sigma of delay costs one sigma of angle.

**Euclidean embedding.** Both terms are linear in their coordinates, so with

```
x_i = [ Omega_i / 2 ,  zeta * (sigma_Omega/sigma_tau) * (tau_i - tau_bar) ]  in R^4
```

`MCD_ij = ||x_i - x_j||` exactly (verified to 1.8e-15). KPowerMeans is then plain
weighted Lloyd at O(NK) — no N x N matrix, ~0.35 s per 2000-path receiver.

### 5.3 KPowerMeans and cluster-count selection

```
J(K)  = sum_i P_i min_k ||x_i - c_k||^2
XB(K) = J(K) / ( sum_i P_i * min_{k!=l} ||c_k - c_l||^2 )
K_hat = argmin XB(K),  K in [2, 12]
```

**Why Xie-Beni and not Calinski-Harabasz.** For a d-dimensional continuum split
into K cells, `J(K) ~ K^(-2/d)` and `d_min ~ K^(-1/d)`, so `XB ~ O(1)` — flat, with
real structure appearing as an interior minimum. `CH = (B/(K-1))/(W/(N-K))` is
monotone increasing on a continuum and returns `K_max` every time. Measured on the
synthetic gate, CH recovers **11.7 clusters** against a truth of 2-4, i.e. it
reproduces the peak-picker's failure mode exactly. Kim-Parks over-counts less
severely (mean bias +1.5). Davies-Bouldin, which averages worst-pairs rather than
taking the strict minimum separation, is better at N=4 and much worse at N=2.

**Parsimony tie-break (`K_TOL = 0.10`).** Ray fans are elongated in the embedding,
so splitting one true cluster often wins the index by a hair. Taking the smallest
K scoring within 10% of the best cuts mean |bias| from 0.71 to 0.45 and RMSE from
1.46 to 0.92 on the synthetic gate. Values above 0.10 change nothing.

**K-free cross-check.** `kernel_power_density` (density peaks on the MCD matrix,
self-tuning bandwidth) needs no K and is reported alongside via the `method`
column. Agreement between two structurally different algorithms is stronger
evidence than either alone. DBSCAN was not used: `sklearn` is absent and `eps` is
as arbitrary as `prominence_db`.

### 5.4 Spec-matched postprocessing

`-25 dB` cluster pruning per Step 6, referenced to the strongest **scattered**
cluster with the LoS ray excluded. TR 38.811 tabulates `numOfCluster` as a
state-dependent constant, so pruning against a K-inflated cluster 1 would make the
count a function of the K-factor, which the reference table is not.

---

## 6. Three estimator corrections

### 6.1 rTau was biased low by (1 - 1/N)

`tau'_n ~ Exp(m)` iid, and the minimum of N iid exponentials is `Exp(m/N)`, so
`E[min] = m/N` and

```
E[ mean_n(tau_n) ] = m - m/N = m (1 - 1/N)
  =>   rTau_hat = mean(tau_k - min tau_k) / ( (1 - 1/N) * sigma_tau )
```

`cluster_stats._r_tau` omits `(1 - 1/N)`, biasing rTau by a factor `1 - 1/N`:
**0.50 at N=2, 0.67 at N=3, 0.75 at N=4** — and N is 2-4 throughout. It compounds a
second error: the denominator used is the spread of the *cluster centroids*, but for
exponential delays `sd(tau') = m` identically, so that ratio is ~1 by construction
and carries no information about rTau. `cluster_id._r_tau_corrected` fixes both;
`ds_source` selects the path-level delay spread from the same path set (`"self"`,
default) or the full-path `tau_rms_mean_s` (`"full_path"`), and both are reported.

`cluster_stats._r_tau` is deliberately left as-is so the cached baseline table
stays reproducible.

### 6.2 cASA/cZSA were in radians

`channel_utils.rms_angular_spread` returns radians;
`cluster_stats.cluster_rx` stored the result in columns named `cASA_deg` /
`cZSA_deg` without conversion. Latent, because the default `store="raw_results"`
carries no angles and every cached value is NaN — but a trap for anyone passing
`store="cluster_samples"`. Fixed in both modules.

### 6.3 cZSA should not use the circular estimator

Zenith is bounded on `[0, pi]` and does not wrap, so `cluster_id` reports a linear
power-weighted spread and carries the circular value as `cZSA_circ_deg` to size the
difference rather than argue it. Measured on real records the two agree closely
(e.g. 15.39 vs 15.44 deg), so this is a correctness fix with little numerical
consequence here — worth knowing, not worth claiming.

### 6.4 perClusterShadowingStd needs pooling

The residual-std estimator needs `N >= 3`, and the corrected N is 2-3, so most
receivers return NaN. `cluster_id.pooled_shadowing_db` fits one residual std to the
`(delay, power)` pairs pooled across all receivers in a (config, elevation, state)
cell, reported as `mu_perClusterShadowingStd_pooled` beside the per-receiver mean.

---

## 7. Validation of the clustering

### 7.1 Synthetic recovery

`cluster_id.synth_38901_channel` generates a Step 5-7 channel with known
parameters, packaged in the `cluster_samples` schema so `cluster_id_rx` consumes it
unchanged. The recoverable truth is `n_eff`, the count **after** Step 6 pruning —
not the drawn N, since the spec's own generation discards clusters below -25 dB.

Clean arm, 27 cells (`N in {2,3,4}` x `rTau in {2.0,2.3,3.5}` x `DS in {10,30,100} ns`),
50 realisations each:

```
mean |bias| = 0.59 clusters      mean RMSE = 1.33
by N_drawn:  N=2  bias +0.30..+1.36     N=3  bias -0.06..-0.32     N=4  bias -0.64..-0.86
cASA recovered  8.9-17.0 deg  (injected 11)
cZSA recovered  5.6- 8.6 deg  (injected  7)
```

**The +/-0.5 gate is met at N=3 and missed at N=2 and N=4.** The bias is not
uniform: the estimator regresses toward ~3 clusters. Under-counting at N=4 comes
from power weighting — a cluster carrying 1% of the power barely moves `J`, so
splitting it off is not rewarded even when it sits above -25 dB. That bias is
*conservative* for the headline claim: it cannot manufacture over-counting.

On the clean arm the peak-picker baseline is also reasonable (mean 2.63 against a
truth of 2.90), because 20 rays per cluster is not a continuum. The difference
appears when the input looks like real RT data.

### 7.2 Diffuse-floor arm — the decisive comparison

Same generator plus 1900 isotropic paths at -30 dB each, giving the ~99%-diffuse
character of the real records (25 realisations per cell):

| N_drawn | n_eff | cluster_id | peak-picker |
|---|---|---|---|
| 2 | 2.00 | 2.88 | **14.00** |
| 3 | 3.00 | 3.36 | **25.60** |
| 4 | 3.96 | 3.28 | **31.44** |

Same ground truth, same records, two methods: one stays within ~1 cluster of truth,
the other tracks the size of the diffuse floor. This is the evidence that the new
method is better and not merely different.

### 7.3 Truncation bound

`cluster_samples` keeps the top 2000 of ~40 000 paths by power. Measured
within-record on the 25 `raw_results` records at each elevation (delay-only form of
the identical pipeline, full path set vs its own top-2000; medians over 25 records):

| elev | N ratio | cDS ratio | DS ratio | discarded power | strongest discarded | mean delay disc/kept (ns) |
|---|---|---|---|---|---|---|
| 10 | 1.000 | 1.000 | 1.000 | 0.5% | -57.9 dB | 347 / 68 |
| 20 | 1.000 | 0.752 | 0.752 | 0.7% | -57.0 dB | 307 / 45 |
| 30 | 1.000 | 0.911 | 0.911 | 1.0% | -56.1 dB | 282 / 35 |
| 40 | 1.000 | 0.617 | 0.617 | 1.2% | -55.8 dB | 263 / 35 |
| 50 | 1.000 | 0.496 | 0.496 | 1.0% | -55.7 dB | 243 / 32 |
| 60 | 1.000 | 0.349 | 0.346 | 1.1% | -55.8 dB | 260 / 34 |
| 70 | 1.000 | 0.246 | 0.246 | 1.2% | -55.6 dB | 258 / 34 |
| 80 | 1.000 | 0.240 | 0.240 | 1.1% | -55.4 dB | 269 / 34 |
| 90 | 1.000 | 0.229 | 0.229 | 1.0% | -55.9 dB | 275 / 34 |

**`numOfCluster` is unbiased by the cut at every elevation** (ratio 1.000
throughout, 225 records). Pruning is max-relative and the strongest discarded path
sits ~56 dB below the peak, so no discarded mass can found a surviving cluster.

**`cDS` is not, and the bias is strongly elevation-dependent** — 1.00 at 10 deg
falling monotonically to 0.23 at 90 deg, median 0.353 overall. Do not quote a
single correction factor: at high elevation the direct path dominates, the
scattered paths are weak and spread far in delay, and the top-2000 keeps a much
narrower delay window than at low elevation where there are fewer paths to begin
with. The per-elevation ratios above are the correction factors; they are reported,
never applied silently.

Caveat: the delay-only form collapses to N=1 on 223 of 225 records, so this check
validates `N` and `cDS`/`DS` but **cannot** bound the rTau bias (only 2 records
yield a finite rTau). That collapse is itself the argument for joint clustering:
without the angular dimension there is no resolvable structure in an RT diffuse
continuum, which is exactly why the delay-only peak-picker has to invent one out of
a prominence threshold.

### 7.4 Method dependence on real data

`cluster_id_sensitivity`, one axis at a time, 40 receivers at 2 GHz / 30 deg. This
is the structural replacement for `cluster_stats.cluster_sensitivity`, whose whole
point was that the peak count is a steep function of `prominence_db`.

| axis | value | N (LOS) | N (NLOS) | rTau (LOS) |
|---|---|---|---|---|
| zeta | 0.25 | 4.03 | 2.88 | 1.34 |
| | 0.5 | 3.93 | 2.80 | 1.62 |
| | **1.0** | **2.78** | **3.05** | **2.27** |
| | 2.0 | 2.00 | 2.33 | 2.61 |
| | 4.0 | 2.00 | 2.13 | 2.76 |
| prune_db | 20 / **25** / 30 | 2.73 / **2.78** / 2.93 | 2.90 / **3.05** / 3.15 | 2.11 / **2.27** / 3.41 |
| k_max | 8 / **12** / 20 | 2.53 / **2.78** / 2.88 | 2.93 / **3.05** / 3.05 | ~2.25 |
| path_dyn_range_db | 30 / **40** / inf | 3.35 / **2.78** / 2.55 | 3.25 / **3.05** / 3.23 | ~2.9 |
| delay_norm | **spread** / czink | **2.78** / 4.45 | **3.05** / 3.10 | **2.27** / 1.33 |
| validity | **xb** / ch / kimparks | **2.78** / 3.95 / 5.28 | **3.05** / 5.68 / 4.45 | ~2.2 |
| method | **kpowermeans** / kpd | **2.78** / 1.05 | **3.05** / 1.23 | 2.27 / 3.59 |

Read honestly, this is a mixed result:

* **Stable where it matters most.** `prune_db` (+/- 5 dB around the spec's 25),
  `k_max` (8 to 20), and `path_dyn_range_db` move N by less than 0.6 clusters. In
  particular there is no ceiling effect — raising `k_max` to 20 does not raise N,
  which is the failure mode CH shows and the peak-picker shows in `prominence_db`.
* **`zeta` matters.** N falls from 4.0 to 2.0 across the swept range, and rTau
  doubles. The delay/angle exchange rate is a real modelling choice, not a
  nuisance parameter: `zeta = 1` is defensible as the isotropic point (one sigma of
  delay costs one sigma of angle) but it is a choice, and N should be quoted as
  "2-4 depending on the delay/angle weighting" rather than as a single number.
* **KPD does not corroborate.** The K-free cross-check returns N ~ 1.05 (LOS) /
  1.23 (NLOS) — it sees a single density mode. The intended argument was that two
  structurally different algorithms agreeing would be strong evidence; they do not
  agree, so that argument is not available. The reading is that the cluster
  structure in this scene is weak enough that a density-based method does not
  resolve it, while a partition-based method with a validity index does. Both
  numbers should be reported.

What survives all of this is the comparison against the baseline: every
configuration in the table returns N between 1 and 5.7, against the peak-picker's
6.6 (LOS) / 11.7 (NLOS) on the same receivers. The method choice moves N by ~2
clusters; the delay-only-vs-joint choice moves it by ~9.

---

## 7.5 Real data, 2 GHz

`cluster_id_table(["results_sieg_2GHz"], range(10,100,10))`, 400 sampled receivers
per elevation split by state. Reference is TR 38.811 Suburban, S band.

| elev | state | n_rx | N | sigma_N | ref N | rTau | ref rTau | cDS ns | ref | zeta_sf | pooled | ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | LOS | 204 | 2.97 | 2.01 | 3 | 3.71 | 2.20 | 7.23 | 1.6 | 4.91 | 7.77 | 3.0 |
| 20 | LOS | 286 | 2.96 | 1.90 | 3 | 2.87 | 3.36 | 6.05 | 1.6 | 3.64 | 5.61 | 3.0 |
| 30 | LOS | 322 | 3.01 | 2.02 | 3 | 2.29 | 3.50 | 5.43 | 1.6 | 3.28 | 4.71 | 3.0 |
| 40 | LOS | 345 | 2.57 | 1.64 | 3 | 2.08 | 2.81 | 6.13 | 1.6 | 3.21 | 3.60 | 3.0 |
| 50 | LOS | 364 | 3.01 | 2.26 | 3 | 2.01 | 2.39 | 6.23 | 1.6 | 2.67 | 3.48 | 3.0 |
| 60 | LOS | 383 | 3.78 | 2.83 | 3 | 2.12 | 2.73 | 6.11 | 1.6 | 3.37 | 4.57 | 3.0 |
| 70 | LOS | 386 | 4.73 | 3.39 | **2** | 2.18 | 2.07 | 5.78 | 1.6 | 3.32 | 4.85 | 3.0 |
| 80 | LOS | 392 | 5.54 | 3.49 | **2** | 2.26 | 2.04 | 5.70 | 1.6 | 3.12 | 5.04 | 3.0 |
| 90 | LOS | 396 | 6.15 | 3.58 | **2** | 2.18 | 2.04 | 5.23 | 1.6 | 3.34 | 5.45 | 3.0 |
| 10 | NLOS | 196 | 3.05 | 1.56 | 4 | 2.97 | 2.30 | 20.88 | 1.6 | 5.76 | 9.07 | 3.0 |
| 20 | NLOS | 114 | 2.89 | 1.45 | 4 | 2.70 | 2.30 | 18.21 | 1.6 | 4.81 | 6.57 | 3.0 |
| 30 | NLOS | 78 | 3.09 | 1.50 | 4 | 2.65 | 2.30 | 15.62 | 1.6 | 3.07 | 4.74 | 3.0 |
| 40 | NLOS | 55 | 3.38 | 1.99 | 4 | 2.38 | 2.30 | 13.26 | 1.6 | 3.71 | 6.10 | 3.0 |
| 50 | NLOS | 36 | 3.44 | 1.42 | 4 | 3.13 | 2.30 | 14.26 | 1.6 | 3.34 | 7.03 | 3.0 |

NLOS above 50 deg holds 4-17 receivers and is withheld by
`validate_3gpp.MIN_N_CLUSTER = 30`. Scored rows (`clusters.csv`, status `ok`):

```
mean |delta|          LOS (9 rows)   NLOS (5 rows)
numOfCluster              1.30           0.83
rTau                      0.60           0.47
perClusterShadowingStd    0.50           1.14
cDS                       4.39          14.85
```

The LOS `numOfCluster` mean is dominated by the four high-elevation rows; over
10-50 deg alone the deltas are -0.03, -0.04, +0.01, -0.43, +0.01.

**numOfCluster.** At 10-50 deg LOS the estimator returns 2.6-3.0 against a
tabulated 3 — the match the whole exercise was for, and a change from the
peak-picker's 6.6. Above 60 deg it rises to 3.8-6.2 while the reference *drops* to
2, and `sigma_N` rises with it (2.0 -> 3.6). At zenith almost every UE is LOS, the
scattered power is a weak diffuse rooftop return, and the index is resolving
structure in something close to noise. Treat LOS N above 60 deg as unreliable and
say so rather than reporting the trend as a finding.

**rTau** is the best-behaved parameter: 2.0-3.7 against a tabulated 2.0-3.5, with
the right order of magnitude at every elevation. Note this is the *corrected*
estimator; the uncorrected form would report these ~33% lower.

**cDS** is 5.2-7.2 ns (LOS) and 12-21 ns (NLOS) against a tabulated 1.6 ns —
over by 3-13x, and the truncation correction makes it worse, not better, since the
measured ratios in 7.3 are all <= 1. The tabulated 1.6 ns is a TR 38.901
terrestrial constant carried into the NTN tables unchanged; RT intra-cluster delay
spread over a real building layout is simply larger.

**perClusterShadowingStd** lands at 2.7-4.9 dB per-receiver against a tabulated
3.0 dB — the closest agreement in the table. The pooled estimator runs higher
(3.5-9.1 dB) because it mixes receivers with different ladders; prefer the
per-receiver mean where `N >= 3` is available and use the pooled value only as a
floor check.

**cASA/cZSA** are reported (25-62 deg / 10-16 deg) but not scorable: the tabulated
11 deg / 7 deg exceed the tabulated *total* ASA/ZSA, as noted in section 4.

---

## 8. Status of each comparison

| comparison | status | note |
|---|---|---|
| DS, ASA, ZSA marginals | scored | S-band zenith reference is internally inconsistent; see `ZENITH_ANOMALY_NOTE` |
| K marginals | scored | `mu_K_dB` is the counterpart to `uK`; ratio-of-means in `rt_alt` |
| clutter loss | scored | new; agrees to 1.4-4.5 dB |
| SF sigma | reported, not scored | scene-limited by the law of total variance |
| LSP correlation matrix | reported | reference is elevation-independent for Suburban |
| decorrelation distance | reported | -8% estimator bias, ~20% scatter, scene-size ceiling |
| numOfCluster, rTau, cDS, perClusterShadowingStd | scored | via `cluster_id`, with the recovery bias above |
| cASA, cZSA | not scorable | reference exceeds the tabulated total spread |
| cASD, raysPerCluster | not scorable | vacuous under NOTE 8 / not an observable |
| ASD, ZSD marginals | vacuous | NOTE 8, agreement true by construction |

---

## 9. Reproducing

```python
import validate_3gpp as v, cluster_id as ci

rt  = v.rt_table(["results_sieg_2GHz"], range(10, 100, 10), labels=["sieg_2GHz"])
rtc = ci.cluster_id_table(["results_sieg_2GHz"], range(10, 100, 10), labels=["sieg_2GHz"])
v.run(["results_sieg_2GHz"], range(10, 100, 10), "validation/2GHz", rt=rt, rtc=rtc)

ci.synth_recovery(n_mc=50)                       # clean gate
ci.synth_recovery(n_mc=25, diffuse_floor=1900)   # diffuse arm
ci.cluster_id_sensitivity(res, n_rx=40)          # method dependence
ci.truncation_bias(res)                          # top-2000 bound
```
