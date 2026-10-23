"""Summary-table values at one elevation from the per-band CSVs (3GPP definition: K and lgDS per UE, then the mean over UEs).
RT rows: P_LoS (metrics*.csv), K (k_3gpp*.csv), K at the UE panel (generator_panel_k*.csv, Ku/Ka), mu_lgDS (ds_3gpp*.csv,
all paths), lgASA, lgZSA, CL, sigma_SF, clusters, sigma_nu (table2_extra*.csv). TR 38.811 rows: post/tr38811_*.csv,
and the ns-3 copy of the 38.811 tables (Ray_Tracing/src/utils/ns3_reference.py) for ASA, ZSA, clusters, CL, sigma_SF;
Ku takes the S-band value (38.811 has no Ku).
Usage: python post/table2.py RUNS [ELEV=30]   ->  RUNS/figures/table2_elevELEV.csv"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ray_Tracing.src.utils import ns3_reference as ns3

runs = sys.argv[1]; e = sys.argv[2] if len(sys.argv) > 2 else "30"; F = f"{runs}/figures"
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [(s, b) for s in ("urban", "rural") for b in ("S", "Ku", "Ka")]


def get(f, col, **eq):
    """Value of col in the row of CSV f at this elevation matching eq; '' if the file or row is missing."""
    if not os.path.exists(f): return ""
    for r in csv.DictReader(l for l in open(f) if not l.startswith("#")):
        if r["elevation_deg"] == e and all(r[k] == v for k, v in eq.items()): return r[col]
    return ""


suf = lambda b: "" if b == "S" else "_" + b
tr = lambda s, b, m: get(f"{KIT}/post/tr38811_{'ka' if b == 'Ka' else 'sband'}.csv", "value", scenario=s, metric=m)
ext = lambda s, b, col, st: get(f"{F}/table2_extra_{s}_{b}.csv", col, state=st)
LSP = {(s, st): ns3.parse_ntn_table(f"NTN{s.capitalize()}{st.upper()}") for s in ("urban", "rural") for st in ("LoS", "NLoS")}
SFCL = {"urban": ns3.parse_sfcl("SFCL_Urban"), "rural": ns3.parse_sfcl("SFCL_SuburbanRural")}
lsp = lambda s, b, st, k: LSP[(s, st)][("Ka" if b == "Ka" else "S", int(e))][k]
sfcl = lambda s, b, k: SFCL[s][int(e)][f"{'Ka' if b == 'Ka' else 'S'}_{k}"]
clu = lambda s, b, st: (f"{ext(s, b, 'clusters_mean', st)} +- {ext(s, b, 'clusters_se', st)} (n {ext(s, b, 'n_raw', st)})"
                        if ext(s, b, "clusters_mean", st) else "")
ROWS = [("P_LoS", "RT", lambda s, b: get(f"{F}/metrics_{s}{suf(b)}.csv", "P_LoS", line="S_VH" if b == "S" else b)),
        ("P_LoS", "TR 38.811", lambda s, b: tr(s, b, "P_LoS")),
        ("K LoS [dB], isotropic", "RT", lambda s, b: get(f"{F}/k_3gpp_{s}{suf(b)}.csv", "mu_K_dB")),
        ("K LoS [dB], isotropic", "TR 38.811", lambda s, b: tr(s, b, "K_LoS_dB")),
        ("K LoS [dB], at the UE panel", "RT", lambda s, b: get(f"{F}/generator_panel_k_{s}_{b}.csv", "K_panel_RT_mean_dB") if b != "S" else ""),
        ("mu_lgDS LoS", "RT", lambda s, b: get(f"{F}/ds_3gpp_{s}{suf(b)}.csv", "mu_lgDS", version="all paths", state="LoS")),
        ("mu_lgDS LoS", "TR 38.811", lambda s, b: tr(s, b, "lgDS_LoS")),
        ("mu_lgDS NLoS", "RT", lambda s, b: get(f"{F}/ds_3gpp_{s}{suf(b)}.csv", "mu_lgDS", version="all paths", state="NLoS")),
        ("mu_lgDS NLoS", "TR 38.811", lambda s, b: tr(s, b, "lgDS_NLoS")),
        *[r for st in ("LoS", "NLoS") for x in ("ASA", "ZSA") for r in (
            (f"mu_lg{x} {st}", "RT", lambda s, b, st=st, x=x: ext(s, b, f"mu_lg{x}", st)),
            (f"mu_lg{x} {st}", "TR 38.811", lambda s, b, st=st, x=x: lsp(s, b, st, f"uLg{x}")))],
        ("CL [dB] (NLoS mean excess loss)", "RT", lambda s, b: ext(s, b, "Lex_mean_dB", "NLoS")),
        ("CL [dB] (NLoS mean excess loss)", "TR 38.811", lambda s, b: sfcl(s, b, "NLOS_CL")),
        ("LoS mean excess loss [dB]", "RT", lambda s, b: ext(s, b, "Lex_mean_dB", "LoS")),
        *[r for st in ("LoS", "NLoS") for r in (
            (f"sigma_SF {st} [dB]", "RT", lambda s, b, st=st: ext(s, b, "Lex_sd_dB", st)),
            (f"sigma_SF {st} [dB]", "TR 38.811", lambda s, b, st=st: sfcl(s, b, f"{st.upper()}_sigF")),
            (f"clusters {st}, mean +- SE", "RT", lambda s, b, st=st: clu(s, b, st)),
            (f"clusters {st}, mean +- SE", "TR 38.811", lambda s, b, st=st: lsp(s, b, st, "numOfCluster")),
            (f"sigma_nu {st} [Hz], mean", "RT", lambda s, b, st=st: ext(s, b, "sigma_nu_mean_Hz", st)),
            (f"sigma_nu {st} [Hz], mean", "TR 38.811", lambda s, b: 0))]]
out = f"{F}/table2_elev{e}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["parameter", "source"] + [f"{s}_{b}" for s, b in COLS])
    for p, src, f in ROWS:
        w.writerow([p, src] + [f(s, b) for s, b in COLS])
print(open(out).read())
