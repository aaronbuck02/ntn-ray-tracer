"""Parse the 3GPP TR 38.811 NTN reference tables out of the ns-3 C++ source.

Pure text parsing -- no built or runnable ns-3 required. Extracts the 22 LSPs
per (band, elevation), the Cholesky factor of the LSP correlation matrix, the
SFCL table and the LoS probability curve.

Five encoding traps are handled here, not by the caller (ZSA precedes ZSD, cDS
is in ns, the NLoS matrix has no K row, the band threshold is a hard 13 GHz
cut, and NOTE 8 zeroes the departure spreads). The S-band zenith columns are
internally inconsistent and should not be trusted -- ZENITH_ANOMALY_NOTE and
zenith_anomaly_check carry that finding. See docs/analysis.md.
"""

from __future__ import annotations

import os
import re
import warnings

import numpy as np

NS3_ROOT = os.environ.get("NS3_ROOT", "")
NS3_CHANNEL_CC = os.path.join(NS3_ROOT, "src/spectrum/model/three-gpp-channel-model.cc")
NS3_PROP_CC = os.path.join(NS3_ROOT, "src/propagation/model/three-gpp-propagation-loss-model.cc")
NS3_CCM_CC = os.path.join(NS3_ROOT, "src/propagation/model/channel-condition-model.cc")

if not NS3_ROOT:
    warnings.warn(
        "NS3_ROOT is not set. Set the NS3_ROOT environment variable to your "
        "ns-3-dev checkout, or pass explicit paths to the parse_* functions.",
        stacklevel=2,
    )

# Exact enum order from three-gpp-channel-model.cc `enum Table3gppParams`.
# NOTE the ZSA-before-ZSD ordering at indices 6-9.
TABLE3GPP_FIELDS = (
    "uLgDS", "sigLgDS",
    "uLgASD", "sigLgASD",
    "uLgASA", "sigLgASA",
    "uLgZSA", "sigLgZSA",     # <-- ZSA first
    "uLgZSD", "sigLgZSD",
    "uK", "sigK",
    "rTau", "uXpr", "sigXpr",
    "numOfCluster", "raysPerCluster",
    "cDS", "cASD", "cASA", "cZSA",
    "perClusterShadowingStd",
)

# LSP orders used by the two Cholesky matrices (see trap 3).
LSP_ORDER_LOS = ("SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA")
LSP_ORDER_NLOS = ("SF", "DS", "ASD", "ASA", "ZSD", "ZSA")

# Columns of SFCL_SuburbanRural, from the enum in three-gpp-propagation-loss-model.cc
SFCL_FIELDS = ("S_LOS_sigF", "S_NLOS_sigF", "S_NLOS_CL",
               "Ka_LOS_sigF", "Ka_NLOS_sigF", "Ka_NLOS_CL")

ZENITH_ANOMALY_NOTE = (
    "TR 38.811 S-band NTN-Suburban zenith spreads as encoded in ns-3 are "
    "internally inconsistent with the Ka-band ones: S has ZSA ~0.02-0.05 deg "
    "with ZSD > ZSA, whereas Ka has ZSA 4-74 deg with ZSD ~0.001 deg. Only the "
    "Ka pattern is physically coherent for a point-source satellite TX. "
    "S-band ZSA/ZSD comparisons should be treated as unreliable."
)


def zenith_anomaly_check(scenario: str = "Suburban", state: str = "LOS") -> dict:
    """
    Quantify the S-vs-Ka zenith inconsistency documented above.

    Returns per-elevation ZSA/ZSD in degrees for both bands plus a
    ``s_band_suspect`` flag (True where the S band reports ZSD >= ZSA, which is
    backwards for a satellite link).
    """
    t = parse_ntn_table(f"NTN{scenario}{state}")
    out = {}
    for (band, elev), v in t.items():
        out.setdefault(elev, {})[band] = (10 ** v["uLgZSA"], 10 ** v["uLgZSD"])
    for elev, d in out.items():
        if "S" in d:
            d["s_band_suspect"] = bool(d["S"][1] >= d["S"][0])
    return dict(sorted(out.items()))


# Transcribed from TR 38.811 Table 6.7.2-5a/6a -- ns-3 does NOT encode these.
TR38811_SUBURBAN_CORR_DIST_M = {
    "LOS":  {"DS": 36, "ASD": 30, "ASA": 25, "SF": 37, "K": 12, "ZSA": 15, "ZSD": 15},
    "NLOS": {"DS": 30, "ASD": 18, "ASA": 15, "SF": 50, "ZSA": 15, "ZSD": 15},
}

# Per-LSP correlation distances, by scenario. ns-3 encodes none of these for any
# scenario (only the *shadowing* one in the propagation model, which
# parse_shadowing_corr_distance does parse), so every entry here is a
# hand-transcribed spec literal and is labelled as such.
#
# DenseUrban/Urban are deliberately EMPTY rather than guessed: they would have
# to come from TR 38.811 Table 6.7.2-1a/2a, and silently reusing the Suburban
# numbers would produce plausible-looking wrong ratios. compare_decorrelation()
# raises on an empty entry. Fill one in and it starts scoring, no other change.
TR38811_CORR_DIST_M = {
    "Suburban":   TR38811_SUBURBAN_CORR_DIST_M,
    "Rural":      TR38811_SUBURBAN_CORR_DIST_M,
    "DenseUrban": {},
    "Urban":      {},
}

# scenario -> (SFCL table in the propagation model, LOS-probability table in the
# channel-condition model). Both are parsed straight out of the ns-3 source.
SCENARIO_TABLES = {
    "DenseUrban": ("SFCL_DenseUrban",    "DenseUrbanLOSProb"),
    "Urban":      ("SFCL_Urban",         "UrbanLOSProb"),
    "Suburban":   ("SFCL_SuburbanRural", "SuburbanRuralLOSProb"),
    "Rural":      ("SFCL_SuburbanRural", "SuburbanRuralLOSProb"),
}


def _read(path: str) -> str:
    with open(path, "r", errors="replace") as fh:
        return fh.read()


def _balanced_block(src: str, anchor: str) -> str:
    """Return the {...} block that follows `anchor`, brace-balanced."""
    i = src.index(anchor)
    j = src.index("{", i)
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j:k + 1]
        k += 1
    raise ValueError(f"unbalanced braces after {anchor!r}")


def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


def parse_ntn_table(name: str = "NTNSuburbanLOS", cc: str = NS3_CHANNEL_CC) -> dict:
    """
    Parse one `std::map<std::string, std::map<int, std::array<float,22>>>`.

    Returns ``{(band, elevation_deg): {field: value}}`` with `field` drawn from
    ``TABLE3GPP_FIELDS``.  Values are floats exactly as written in the source.
    """
    src = _read(cc)
    block = _strip_comments(_balanced_block(src, f"> {name}"))
    # clang-format wraps entries across arbitrary line breaks; collapsing all
    # whitespace makes the wrapping irrelevant to the regexes below.
    flat = re.sub(r"\s+", " ", block)

    out = {}
    # Split into per-band sections, keeping the band label.
    parts = re.split(r'\{\s*"(S|Ka)"\s*,', flat)
    if len(parts) < 3:
        raise ValueError(f"no band sections found in {name}")
    for bi in range(1, len(parts), 2):
        band, section = parts[bi], parts[bi + 1]
        for m in re.finditer(r"\{\s*(\d+)\s*,\s*\{([^{}]*)\}\s*\}", section):
            elev = int(m.group(1))
            vals = [float(v) for v in m.group(2).split(",") if v.strip()]
            if len(vals) != 22:
                raise ValueError(
                    f"{name} {band} {elev}deg: expected 22 values, got {len(vals)}")
            out[(band, elev)] = dict(zip(TABLE3GPP_FIELDS, vals))
    if not out:
        raise ValueError(f"parsed no rows from {name}")
    return out


def parse_sqrtc(name: str = "sqrtC_NTN_Suburban_LOS", cc: str = NS3_CHANNEL_CC) -> np.ndarray:
    """Parse a `constexpr std::array<std::array<double,N>,N>` lower-triangular L."""
    src = _read(cc)
    block = _strip_comments(_balanced_block(src, f"> {name}"))
    flat = re.sub(r"\s+", " ", block)
    rows = [[float(v) for v in m.group(1).split(",") if v.strip()]
            for m in re.finditer(r"\{([^{}]*)\}", flat)]
    rows = [r for r in rows if r]
    L = np.array(rows, dtype=float)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"{name}: parsed shape {L.shape}, expected square")
    return L


def correlation_from_sqrt(L: np.ndarray) -> np.ndarray:
    """C = L @ L.T -- recover the LSP cross-correlation matrix."""
    return L @ L.T


def parse_sfcl(name: str = "SFCL_SuburbanRural", cc: str = NS3_PROP_CC) -> dict:
    """Parse the shadow-fading-sigma / clutter-loss map -> {elev: {field: val}}."""
    src = _read(cc)
    block = _strip_comments(_balanced_block(src, name))
    flat = re.sub(r"\s+", " ", block)
    out = {}
    for m in re.finditer(r"\{\s*(\d+)\s*,\s*\{([^{}]*)\}\s*\}", flat):
        vals = [float(v) for v in m.group(2).split(",") if v.strip()]
        out[int(m.group(1))] = dict(zip(SFCL_FIELDS, vals))
    if not out:
        raise ValueError(f"parsed no rows from {name}")
    return out


def parse_los_prob(name: str = "SuburbanRuralLOSProb", cc: str = NS3_CCM_CC) -> dict:
    """Parse the suburban/rural LOS-probability table -> {elev: prob}."""
    src = _read(cc)
    block = _strip_comments(_balanced_block(src, name))
    flat = re.sub(r"\s+", " ", block)
    # ns-3 writes these as `{10, {0.782}}` (brace-wrapped scalar); older/other
    # tables use the bare `{10, 0.782}` form. Accept both.
    out = {int(m.group(1)): float(m.group(2))
           for m in re.finditer(r"\{\s*(\d+)\s*,\s*\{?\s*([0-9.eE+-]+)\s*\}?\s*\}", flat)}
    if not out:
        raise ValueError(f"parsed no rows from {name}")
    return out


def parse_shadowing_corr_distance(scenario: str = "Suburban",
                                  cc: str = NS3_PROP_CC) -> dict:
    """Shadowing decorrelation distance for an NTN model, in metres."""
    src = _read(cc)
    anchor = (f"ThreeGppNTN{scenario}PropagationLossModel::"
              f"GetShadowingCorrelationDistance")
    if anchor not in src:
        raise ValueError(f"{anchor} not found in {cc}")
    body = _strip_comments(_balanced_block(src, anchor))
    nums = [float(x) for x in
            re.findall(r"correlationDistance\s*=\s*([0-9.]+)\s*;", body)]
    if len(nums) < 2:
        raise ValueError(
            f"expected 2 correlationDistance assignments in {anchor}, got {nums}")
    return {"LOS_m": nums[0], "NLOS_m": nums[1]}


def band_for_frequency(f_hz: float, warn: bool = True) -> str:
    """
    Mirror ns-3's band selection: `fcGHz < 13 ? "S" : "Ka"`.

    There is no interpolation and no Ku-band table in TR 38.811, so an 11 GHz
    carrier is silently scored against S-band values measured near 2 GHz.
    """
    band = "S" if f_hz < 13e9 else "Ka"
    if warn and 3e9 < f_hz < 13e9:
        warnings.warn(
            f"{f_hz/1e9:.1f} GHz falls between the TR 38.811 S (~2 GHz) and "
            f"Ka (~20-30 GHz) bands; ns-3 maps it to '{band}'. Treat the "
            f"comparison as bracketing, not like-for-like.",
            stacklevel=2)
    return band


def apply_satellite_note8(row: dict) -> dict:
    """
    TR 38.811 NOTE 8 / ns-3 satellite override: for a spaceborne transmitter the
    departure spreads collapse, because the TX subtends a negligible angle.
    ns-3 sets uLgASD/uLgZSD to ~0 with zero sigma above 50 km altitude.
    """
    r = dict(row)
    r["uLgASD"] = -np.inf
    r["sigLgASD"] = 0.0
    r["uLgZSD"] = -np.inf
    r["sigLgZSD"] = 0.0
    r["note8_applied"] = True
    return r


def reference_frame(scenario: str = "Suburban",
                    satellite_note8: bool = True,
                    cc: str = NS3_CHANNEL_CC):
    """
    Tidy long-form table of the TR 38.811 reference values.

    One row per (state, band, elevation); columns are the 22 LSP fields with
    ``cDS_s`` added in seconds, plus ``sigma_SF_dB``, ``clutter_loss_dB`` and
    ``los_prob`` merged in from the propagation/channel-condition modules.
    """
    import pandas as pd

    if scenario not in SCENARIO_TABLES:
        raise ValueError(f"unknown scenario {scenario!r}; "
                         f"expected one of {sorted(SCENARIO_TABLES)}")
    sfcl_name, losp_name = SCENARIO_TABLES[scenario]
    sfcl = parse_sfcl(sfcl_name)
    losp = parse_los_prob(losp_name)

    rows = []
    for state in ("LOS", "NLOS"):
        tbl = parse_ntn_table(f"NTN{scenario}{state}", cc=cc)
        for (band, elev), vals in sorted(tbl.items()):
            r = apply_satellite_note8(vals) if satellite_note8 else dict(vals)
            r.setdefault("note8_applied", False)
            r["scenario"], r["state"], r["band"], r["elevation_deg"] = \
                scenario, state, band, elev
            r["cDS_s"] = vals["cDS"] * 1e-9          # source stores ns
            if sfcl and elev in sfcl:
                s = sfcl[elev]
                r["sigma_SF_dB"] = s[f"{band}_{state}_sigF"]
                r["clutter_loss_dB"] = s.get(f"{band}_NLOS_CL") if state == "NLOS" else 0.0
            if losp and elev in losp:
                r["los_prob"] = losp[elev]
            rows.append(r)

    df = pd.DataFrame(rows)
    lead = ["scenario", "state", "band", "elevation_deg"]
    return df[lead + [c for c in df.columns if c not in lead]]


def correlation_frame(scenario: str = "Suburban", state: str = "LOS"):
    """Correlation matrix as a labelled DataFrame, in ns-3's own LSP order."""
    import pandas as pd
    L = parse_sqrtc(f"sqrtC_NTN_{scenario}_{state}")
    C = correlation_from_sqrt(L)
    order = LSP_ORDER_LOS if state == "LOS" else LSP_ORDER_NLOS
    if C.shape[0] != len(order):
        raise ValueError(f"{state}: matrix is {C.shape[0]}x{C.shape[0]} "
                         f"but expected {len(order)} ({order})")
    return pd.DataFrame(np.round(C, 6), index=list(order), columns=list(order))


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 200)
    ref = reference_frame("Suburban")
    print("TR 38.811 NTN-Suburban reference, parsed from ns-3")
    print(ref.groupby(["state", "band"]).size(), "\n")
    cols = ["elevation_deg", "uLgDS", "sigLgDS", "uLgASA", "sigLgASA",
            "uLgZSA", "sigLgZSA", "uK", "sigK", "numOfCluster",
            "sigma_SF_dB", "los_prob"]
    print(ref[(ref.state == "LOS") & (ref.band == "S")][cols].to_string(index=False))
    print("\nLOS correlation matrix (C = L @ L.T):")
    print(correlation_frame("Suburban", "LOS").to_string())
