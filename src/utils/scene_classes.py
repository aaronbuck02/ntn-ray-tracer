"""NTN scene classes (rural / suburban / urban) for the scene-classification dataset.

Class medians map Chiu & Roy's wall process onto the P.1410 block sampler; each
scene perturbs them, draws class-weighted ITU materials per role and a
class-blind polarization pair. See docs/scene_classification.md.
"""

import contextlib
import copy
import io

import numpy as np
from scipy.special import erfc

from Ray_Tracing.src.utils import scene_gen_ppp as sg
from Ray_Tracing.src.utils import variant_manifest as vm

CLASSES = ("rural", "suburban", "urban")

# (gamma [m], lambda_1D [1/m], l_f [m]): Chiu & Roy Sec. VI rural/urban; suburban interpolates
# (lambda_1D * gamma near the geometric mean of the other two).
CHIU_WALL = {"rural": (3.0, 0.020, 10.0), "suburban": (5.0, 0.022, 15.0), "urban": (8.0, 0.025, 20.0)}

# Blocks with streets block more than zero-width walls at the same lambda_1D (square
# mean width 4L/pi, UEs next to buildings). Per-class density scale fitted so block
# P_LoS matches Chiu eq. (27) within 3.5 pts at 10-80 deg (calibration_table).
KAPPA = {"rural": 0.72, "suburban": 0.66, "urban": 0.60}


def block_medians(gamma, lam_1d, l_f, kappa):
    """P.1410 medians: footprint side l_f, crossing rate kappa * lambda_1D."""
    return dict(alpha=kappa * lam_1d * l_f, beta=kappa * lam_1d / l_f * 1e6, gamma=gamma)


CLASS_MEDIANS = {c: block_medians(*CHIU_WALL[c], KAPPA[c]) for c in CLASSES}
PERTURB = 0.20                # geo factors log-uniform on median x/÷ (1 + PERTURB)
SEED_BLOCK = {"rural": 0, "suburban": 100_000, "urban": 200_000}

# alpha/gamma/cell_res are per-scene draws. h_min 0.5 m: Chiu is untruncated, but
# near-zero boxes would sit coplanar with the ground.
PPP_KW = dict(Lx=500.0, Ly=500.0, cell_res=40.0, alpha=0.3, points_per_cell=4,
              gamma=5.0, h_min=0.5, h_max=120.0, max_fill=0.8, min_side=3.0)
PPP_MATERIALS = dict(building_material="itu_concrete", roof_material="itu_brick",
                     ground_material="itu_plasterboard")   # dummy identities, one per role

ANCHORS = {"ground": ("concrete", "medium_dry_ground", "wet_ground"),
           "roof":   ("metal", "concrete", "brick", "wood"),
           "wall":   ("concrete", "glass", "brick", "wood")}
MATERIAL_WEIGHTS = {
    "urban":    {"ground": (0.80, 0.12, 0.08), "roof": (0.60, 0.30, 0.10, 0.00),
                 "wall": (0.65, 0.25, 0.10, 0.00)},
    "suburban": {"ground": (0.35, 0.35, 0.30), "roof": (0.10, 0.05, 0.25, 0.60),
                 "wall": (0.15, 0.00, 0.40, 0.45)},
    "rural":    {"ground": (0.05, 0.50, 0.45), "roof": (0.15, 0.00, 0.15, 0.70),
                 "wall": (0.10, 0.00, 0.30, 0.60)},
}
POL_PAIRS = ("RHCP/RHCP", "RHCP/LHCP", "V/V", "H/H", "RHCP/V")
LINK_FACTORS = {"cfg.POL_PAIR": (POL_PAIRS, None, "cat")}


def anchor_weights(cls, role, pooled=False):
    if pooled:   # material-control ablation: same mixture for every class
        return tuple(np.mean([MATERIAL_WEIGHTS[c][role] for c in CLASSES], axis=0))
    return MATERIAL_WEIGHTS[cls][role]


def geo_factors(cls, perturb=PERTURB):
    m, f = CLASS_MEDIANS[cls], 1.0 + perturb
    return {**{f"geo.{k}": (m[k] / f, m[k] * f, "log") for k in ("alpha", "beta", "gamma")},
            "cfg.SCENE_EAST_HEADING_DEG": (0.0, 90.0)}


def material_factors(cls, pooled=False):
    out = {}
    for r in vm.ROLES:
        out[f"{r}.base"] = (ANCHORS[r], anchor_weights(cls, r, pooled), "cat")
        out[f"{r}.eps_scale"] = (0.8, 1.25, "log")
        out[f"{r}.sigma_scale"] = (0.8, 1.25, "log")
        out[f"{r}.scattering_coefficient"] = (0.0, 0.6)
    out["all.alpha_r"] = (1, 10, "int")
    out["all.xpd_coefficient"] = (0.0, 0.5)
    return out


def class_factors(cls, perturb=PERTURB, pooled=False):
    return {**geo_factors(cls, perturb), **material_factors(cls, pooled), **LINK_FACTORS}


def class_seeds(cls, n_scenes, seed_offset=0):
    b = SEED_BLOCK[cls] + seed_offset
    return range(b, b + n_scenes)


def class_manifest(cls, n_scenes, table, perturb=PERTURB, pooled=False, seed_offset=0):
    """Streaming manifest of one class: per-scene LHS draw of class_factors, real orbits."""
    base_seed = SEED_BLOCK[cls] + seed_offset
    m = vm.virtual_manifest(n_scenes, PPP_KW, base_seed=base_seed, **PPP_MATERIALS)
    m = vm.lhs_manifest(m, class_factors(cls, perturb, pooled), seed=base_seed + 1)
    m = vm.attach_orbits(m, table)
    m["scene_class"] = cls
    m["class_spec"] = dict(perturb=perturb, pooled=pooled, seed_offset=seed_offset,
                           medians=CLASS_MEDIANS[cls], kappa=KAPPA[cls])
    for e in m["scenes"]:
        e["scene_class"] = cls
    return m


def fixed_pol_manifest(m, pair):
    """Copy of manifest m with every scene's polarization pair set to pair ("TX/RX"); all else kept."""
    tx, rx = pair.split("/")
    m = copy.deepcopy(m)
    for e in m["scenes"]:
        e["cfg_overrides"].update(TX_POL_TYPE=tx, RX_POL_TYPE=rx)
    m["class_spec"]["fixed_pol"] = pair
    return m


# ─── Calibration gate (CPU) ───────────────────────────────────────────────────

def plos_chiu(elev_deg, cls, h_g=1.5):
    """Chiu & Roy eq. (27) P_LoS for the class's wall parameters."""
    gamma, lam_1d, _ = CHIU_WALL[cls]
    b = np.radians(np.asarray(elev_deg, dtype=float))
    return np.exp(-np.sqrt(np.pi / 2) * lam_1d * gamma / np.tan(b)
                  * erfc(h_g / (np.sqrt(2) * gamma)))


def plos_block_mc(spec, elev_deg, n_ue, rng, h_g=1.5):
    """LoS fraction over n_ue free-space UEs, uniform azimuth: ray / axis-aligned box test."""
    with contextlib.redirect_stdout(io.StringIO()):
        free, _ = sg.free_space_grid(spec, 2.0, 20.0, 1.0)
    ue = free[rng.choice(len(free), n_ue)]
    az = rng.uniform(0.0, 2 * np.pi, n_ue)
    c, s, H = spec.centers, spec.sides / 2, spec.heights
    tb, clear = np.tan(np.radians(elev_deg)), 0
    for (x, y), a in zip(ue, az):
        d = (np.cos(a), np.sin(a))
        with np.errstate(divide="ignore", invalid="ignore"):
            tx1, tx2 = (c[:, 0] - s - x) / d[0], (c[:, 0] + s - x) / d[0]
            ty1, ty2 = (c[:, 1] - s - y) / d[1], (c[:, 1] + s - y) / d[1]
        t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        t_out = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
        t0 = np.maximum(t_in, 0.0)
        clear += not np.any((t_out > t0) & (h_g + t0 * tb < H))   # lowest ray point is the entry
    return clear / n_ue


def calibration_table(elevations=(10, 20, 30, 40, 50, 60, 70, 80), n_scenes=16, n_ue=150, seed=0):
    """Rows per (class, elevation): block P_LoS at the medians vs Chiu, plus realised alpha/beta/gamma."""
    rows = []
    for cls in CLASSES:
        rng = np.random.default_rng(seed)
        kw = vm._geo_kwargs(PPP_KW, CLASS_MEDIANS[cls])
        specs = [sg.sample_ppp_scene(**kw, seed=10_000 + i) for i in range(n_scenes)]
        real = np.mean([[s.meta["alpha_realised"], s.meta["itu_beta"], s.meta["itu_gamma_fit"]]
                        for s in specs], axis=0)
        for e in elevations:
            p = np.mean([plos_block_mc(s, e, n_ue, rng) for s in specs])
            rows.append(dict(scene_class=cls, elevation=e, P_los_block=p,
                             P_los_chiu=float(plos_chiu(e, cls)), alpha_real=real[0],
                             beta_real=real[1], gamma_real=real[2]))
    return rows
