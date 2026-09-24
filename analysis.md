# Analysis, validation and export

How ray-tracing output becomes 3GPP-comparable tables. The per-run validation
procedures and their acceptance criteria live in
[`../validation/METHODS.md`](../validation/METHODS.md); this file covers the
modules.

| Module | Role |
|---|---|
| `src/utils/lsp_stats.py` | per-RX output → TR 38.811-style LSP table |
| `src/utils/ns3_reference.py` | parse the TR 38.811 reference tables out of ns-3 C++ source |
| `src/utils/validate_3gpp.py` | join the two, score the deltas |
| `src/utils/cluster_stats.py` | delay-cluster and PDP peak extraction |
| `src/utils/cluster_id.py` | joint delay–angle clustering (KPowerMeans in MCD space) |

---

## LSP distillation — `lsp_stats.py`

Turns the per-UE quantities the runners save (K, DS, ASD/ASA/ZSD/ZSA,
per-mechanism powers, UE positions) into exactly what 3GPP tabulates, so the two
compare row for row: μ/σ of `log10(DS / 1 s)` and `log10(spread / 1 deg)`, mean
and std of per-link K in dB, shadow fading as a zero-mean residual, the LSP
cross-correlation matrix in ns-3's LSP orders, and the spatial decorrelation
distance per LSP.

### Why the definitions are what they are

**Loss** reuses `elevation_stats_to_ns3`'s convention exactly
(`-10*log10(P_los + P_nlos)`) so the two tables can never disagree.

**SF is a zero-mean residual** about the per-(elevation, state) mean, not a raw
standard deviation. 3GPP's SF is a residual after removing distance-dependent
path loss and mean clutter loss; expressing it that way is what makes it
admissible as a *column of the correlation matrix*. Over a ~340 m campus at LEO
slant range the geometric range spread is negligible, so the state mean is an
adequate stand-in for a fitted path-loss model.

**K has three estimators, and they are not interchangeable:**

| Estimator | Definition | Use |
|---|---|---|
| `mu_K_dB` / `sigma_K_dB` | mean and std of *per-link* dB over the state subset | what TR 38.811's `uK`/`sigK` are defined as; `K_dB` is the per-receiver column feeding the correlation matrix and the K decorrelation distance |
| `K_dB_ratio_of_means` | `E[P_los]/E[P_nlos]` over the state subset | continuity with `elevation_stats_to_ns3` / `plot_utils`; the μ the report tabulates |
| `K_dB_ratio_all` | the same ratio over *every* receiver (NLoS contribute `P_los = 0`) | whole-scene diagnostic printed per batch as `K_mean`; not what the report tabulates |

The correlation matrix and decorrelation distance need one value per receiver,
so neither ratio form can replace the per-link one.

The ratio forms are power-weighted, so they are set by the receivers holding the
most NLoS power rather than by the typical link. Where per-link K is broadly
spread they diverge sharply from `mu_K_dB` — **~5 dB at 10° on the UW campus**,
where the LoS population is bimodal — and converge at zenith. The report pairs
`K_dB_ratio_of_means` as μ with `sigma_K_dB` as σ; both are restricted to the
LoS subset, but they remain estimators of *different quantities*, and the report
text says so.

**Degenerate ASD/ZSD**: for a satellite link the departure spreads are ~0 by
geometry, so their `log10` is meaningless. Those cells report `mu = -inf,
sigma = 0` with a note — a **match** to TR 38.811 NOTE 8, not a failure.

### Solver Monte-Carlo noise

Sionna's `PathSolver` samples the diffuse component stochastically and is **not**
seeded from `sim_config.json`, so two runs over an identical scene, receiver set
and config do not reproduce bit for bit. Measured on the UW campus (200 UEs,
30°, 1e6 samples, two consecutive runs):

| Quantity | Change |
|---|---|
| `los_probability`, `P_los` | identical (LoS is geometric) |
| median DS | 0.01 % |
| median K | 0.14 % |
| median ASA | 0.22 % |
| per-UE DS | median 0.00 %, p95 0.05 %, max 21 % |
| per-UE K | median 0.00 dB, p95 0.005 dB, max 9.9 dB |

**Aggregate** LSP statistics are stable to a few tenths of a percent, far below
the RT-vs-38.811 differences they assess. An **individual** receiver's
NLoS-dominated K or DS is not reproducible — one strong diffuse path entering or
leaving the sample set moves it. Do not quote per-UE values as exact.

---

## Reference tables — `ns3_reference.py`

Extracts the TR 38.811 NTN tables straight out of the ns-3 C++ source, so the
RT table is compared against published values without hand-transcription. ns-3
encodes TR 38.811 V15.4.0 as static C++ tables; these are pure text parses and
need no built or runnable ns-3.

Extracted: `NTNSuburbanLOS`/`NLOS` (and Urban/Rural) — 22 LSPs per
(band, elevation); `sqrtC_NTN_Suburban_LOS`/`_NLOS`, the Cholesky square root of
the LSP cross-correlation matrix, recovered as `C = L @ L.T` (which returns
exact TR 38.811 round numbers and is therefore also a strong parser self-check);
`SFCL_SuburbanRural` shadow-fading σ and NLoS clutter loss; and LoS probability
vs elevation.

### Traps this module handles

1. **ZSA precedes ZSD** in the 22-value array (indices 6–9 are `uLgZSA`,
   `sigLgZSA`, `uLgZSD`, `sigLgZSD`). Reading them in the "natural"
   ASD/ASA/ZSD/ZSA order silently swaps the two zenith spreads.
2. **`cDS` is stored in nanoseconds** and multiplied by 1e-9 at use in ns-3.
   `reference_frame` exposes `cDS_s` in seconds.
3. **The NLoS correlation matrix has no K row/column**: its order is
   `[SF, DS, ASD, ASA, ZSD, ZSA]` (6×6) against LoS's
   `[SF, K, DS, ASD, ASA, ZSD, ZSA]` (7×7).
4. **Band selection is a hard threshold at 13 GHz** (`fcGHz < 13 ? "S" : "Ka"`).
   An 11 GHz (Ku) carrier is therefore scored against the S-band table, whose
   values were measured at ~2 GHz. `band_for_frequency` reproduces the rule and
   warns — it is the single largest source of legitimate RT-vs-38.811
   disagreement in a Ku study.
5. **TR 38.811 NOTE 8**: above 50 km, ns-3 overwrites the departure spreads
   (ASD, ZSD) with ~0. `apply_satellite_note8` reproduces this and is on by
   default, since both endpoints of an NTN link qualify.

### Known anomaly — the S-band zenith columns

As encoded in ns-3, the NTN-Suburban LoS zenith spreads are internally
inconsistent between bands:

| Elevation | S: ZSA | S: ZSD | Ka: ZSA | Ka: ZSD |
|---|---|---|---|---|
| 30° | 0.021° | 0.053° | 12.88° | 0.0009° |
| 60° | 0.031° | 0.044° | 44.67° | 0.0014° |

The Ka column is physically coherent for a satellite link: a real arrival spread
that grows with elevation, and a departure spread of essentially zero because
the transmitter subtends no angle. The S column has *both* at ~0, and ZSD
**larger** than ZSA — the point-source satellite spreading more in departure
than the ground clutter does in arrival, which is backwards.

**Treat S-band ZSA/ZSD comparisons as unreliable.** This matters directly for a
Ku study, because `band_for_frequency` maps 11 GHz to "S" while the Ka column is
the more physically meaningful arrival-zenith reference. `ZENITH_ANOMALY_NOTE`
carries this text for inclusion in reports, and `zenith_anomaly_check` flags the
affected rows.

### Not extractable

ns-3 does not encode per-LSP spatial correlation distances anywhere. The only
decorrelation distance in the source is the *shadowing* one in the propagation
loss model (LoS 37 m / NLoS 50 m), which `parse_shadowing_corr_distance` does
parse. The per-LSP values in `TR38811_SUBURBAN_CORR_DIST_M` are transcribed
literals from the spec and are labelled as such rather than pretending to be
parsed.

---

## Clustering — `cluster_stats.py` and `cluster_id.py`

TR 38.811 tabulates cluster parameters that TR 38.901 Sec 7.5 *generates*: the
spec defines no procedure for identifying a cluster **in data**. So there are
two estimators here, and they disagree by design.

### `cluster_stats.py` — delay-only peak picking

Picks peaks of the binned power delay profile. Simple and fast, but it is
delay-only and counts prominences of a diffuse continuum, so it returns
**6–12 clusters against a tabulated 3–4**. Its `ensemble_pdp` and
`pdp_delay_spread_ns` are the PDP-side view of the same quantity `lsp_stats`
reports as DS, which makes the two a useful internal consistency check.

Entry points: `cluster_rx`, `cluster_elevation`, `cluster_table`,
`cluster_sensitivity`.

### `cluster_id.py` — joint delay–angle clustering

Uses the estimator family the measurement literature actually used to produce
those tables: power-weighted KPowerMeans over the Multipath Component Distance
(Steinbauer; Czink), with the cluster count chosen by a validity index and the
spec's own −25 dB pruning applied afterwards.

Four validity indices are available (`xie_beni`, `calinski_harabasz`,
`davies_bouldin`, `kim_parks`) behind `select_k`. It consumes `cluster_samples`
— the only store carrying per-path angles — and emits the column names
`validate_3gpp.compare_clusters` reads, so it is a drop-in replacement for
`cluster_stats.cluster_table`.

Departure angles are dropped: measured spread within one receiver is 0.0006° in
zenith and 0.002° in azimuth, which is TR 38.811 NOTE 8 showing up as data.

`synth_38901_channel` / `synth_recovery` generate a channel with known cluster
structure and check the estimator recovers it — the validation gate for this
module. `cluster_id_sensitivity` sweeps the hyperparameters.

---

## Scoring — `validate_3gpp.py`

Joins the RT table (`lsp_stats`) against the reference (`ns3_reference`) and
scores the deltas. `run` drives the whole comparison; the individual
comparisons are available separately:

| Function | Compares |
|---|---|
| `compare_marginals` | per-LSP μ and σ against the tabulated values |
| `compare_clusters` | cluster count, `r_tau`, `c_DS`, per-cluster shadowing |
| `compare_correlation` | the LSP cross-correlation matrix |
| `compare_decorrelation` | per-LSP spatial decorrelation distances |
| `compare_clutter_loss` | NLoS clutter loss |
| `compare_los_probability` | the geometric LoS fraction against the tabulated curve |

Cells that are vacuous under TR 38.811 NOTE 8 — the departure spreads, which
both sides force to −∞ — are reported as matches by construction rather than
scored, so they cannot inflate an agreement count.

Acceptance criteria per run live in
[`../validation/METHODS.md`](../validation/METHODS.md).
