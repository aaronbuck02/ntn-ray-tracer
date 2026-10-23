import pickle, numpy as np

def load_s_band(runs, scen, el):
    """S-band V and H results of one elevation, combined per UE as the sum over the two ports."""
    v, h = (pickle.load(open(f"{runs}/{scen}_S_{p}/results_elev{el:03d}deg.pkl", "rb")) for p in "VH")
    assert v["scene_seeds"] == h["scene_seeds"] and v["nominal_rx_positions"] == h["nominal_rx_positions"]
    g = lambda r, k: np.asarray(r[k], float)
    out = {k: g(v, k) + g(h, k) for k in ("P_los", "P_nlos", "P_nlos_2us")}
    wv, wh = g(v, "P_los") + g(v, "P_nlos_2us"), g(h, "P_los") + g(h, "P_nlos_2us")
    w = wv + wh
    mv, mh = g(v, "nu_mean_Hz"), g(h, "nu_mean_Hz")
    mu = (wv * mv + wh * mh) / w
    out["nu_mean_Hz"] = mu
    out["sigma_nu_Hz"] = np.sqrt((wv * (g(v, "sigma_nu_Hz") ** 2 + (mv - mu) ** 2)
                                  + wh * (g(h, "sigma_nu_Hz") ** 2 + (mh - mu) ** 2)) / w)
    # per-port mean delays are not saved, so this drops their difference; median error <0.1% (raw-sample check)
    out["DS_ns"] = np.sqrt((wv * g(v, "DS_ns") ** 2 + wh * g(h, "DS_ns") ** 2) / w)
    out["los"] = g(v, "los_probability")
    out["K_dB"] = 10 * np.log10(out["P_los"] / out["P_nlos"])
    return out

if __name__ == "__main__":
    import sys
    o = load_s_band(sys.argv[1], sys.argv[2], int(sys.argv[3]))   # e.g. runs urban 30
    m = o["los"] == 1
    print("P_LOS %.3f | K median (LOS) %.1f dB | DS median %.1f ns | sigma_nu median %.2f Hz"
          % (m.mean(), np.median(o["K_dB"][m]), np.nanmedian(o["DS_ns"]), np.nanmedian(o["sigma_nu_Hz"])))
