"""GPU-free checks for pass geometry, coherent SNR and bin denominators."""

import numpy as np
import pytest

from Ray_Tracing.src.utils.pass_snr import (elevation_bin_statistics,
                                           make_target_pass, make_real_target_passes,
                                           snr_db_from_cir, wideband_snr_db)


# Real elements preserved from the repository's 2026-08-24 Starlink archive.
REAL_TLES = """STARLINK-36171
1 67027U 25293AA  26236.40449642  .00002456  00000+0  21352-3 0  9997
2 67027  70.0017  31.4339 0002584 268.1313  91.9550 14.98337176 40018
STARLINK-2047
1 47355U 21005G   26235.60201996  .00001986  00000+0  75715-4 0  9995
2 47355  53.1608 314.2608 0000953  87.7430 272.3681 15.31710585309321
STARLINK-36065
1 66898U 25283W   26235.73701283  .00028369  00000+0  84743-3 0  9992
2 66898  53.1590 325.1473 0002999  85.8201 274.3145 15.34403972 41469
"""


def test_real_tle_selection_and_unchanged_geometry():
    from datetime import datetime
    from sgp4.api import jday
    from Ray_Tracing.src.utils.orbital_utils import load_tle_satrec, propagate_topocentric

    latitude, longitude = 51.504646101082955, -0.019568475850757397
    passes = make_real_target_passes(REAL_TLES, [30, 60, 90], latitude, longitude,
                                    epoch_utc="2026-08-24T05:29:53.573427Z",
                                    search_hours=3, tolerance_deg=0.2)
    assert [p["metadata"]["norad_id"] for p in passes] == [67027, 47355, 66898]
    assert [p["metadata"]["inclination_deg"] for p in passes] == pytest.approx([70.0017, 53.1608, 53.1590])
    for p in passes:
        meta, rows = p["metadata"], p["waypoints"]
        assert meta["tle_line1"] in REAL_TLES and meta["tle_line2"] in REAL_TLES
        assert meta["orbit_source"] == "real catalog TLE propagated with SGP4"
        sat = load_tle_satrec(meta["tle_line1"], meta["tle_line2"])
        start = datetime.fromisoformat(meta["epoch_utc"])
        jd, fr = jday(start.year, start.month, start.day, start.hour, start.minute,
                      start.second + start.microsecond / 1e6)
        for w in rows[::10]:
            actual = propagate_topocentric(sat, jd, fr + w["epoch_offset_s"] / 86400,
                                            latitude, longitude, 0)
            assert actual[1] == pytest.approx(w["elev_deg"], abs=2e-5)
            assert actual[2] * 1e3 == pytest.approx(w["range_m"], abs=0.1)
        assert {w["leg"] for w in rows} == {"rising", "setting"}
        assert len(rows) * meta["sample_interval_s"] == pytest.approx(meta["duration_s"])
        for t in (meta["rise_epoch_offset_s"], meta["set_epoch_offset_s"]):
            el = propagate_topocentric(sat, jd, fr + t / 86400, latitude, longitude, 0)[1]
            assert el == pytest.approx(0, abs=1e-5)
    assert 89 < passes[-1]["metadata"]["actual_max_elevation_deg"] < 90


def test_real_search_rejects_stale_elements():
    with pytest.raises(RuntimeError, match="No qualifying real passes"):
        make_real_target_passes(REAL_TLES, [30], 51.5, -0.02,
                                epoch_utc="2026-10-06T00:00:00Z")


def test_real_selection_respects_azimuth_and_excluded_satellites():
    kwargs = dict(epoch_utc="2026-08-24T05:29:53.573427Z", search_hours=3,
                  tolerance_deg=0.2, azimuth_tolerance_deg=1)
    initial = make_real_target_passes(REAL_TLES, [30], 51.504646101082955,
                                     -0.019568475850757397, **kwargs)[0]
    az = initial["metadata"]["peak_azimuth_deg"]
    selected = make_real_target_passes(REAL_TLES, [30], 51.504646101082955,
                                      -0.019568475850757397, target_azimuths=[az], **kwargs)
    assert selected[0]["metadata"]["norad_id"] == 67027
    assert selected[0]["metadata"]["target_peak_azimuth_deg"] == az
    with pytest.raises(RuntimeError, match="No distinct real satellite"):
        make_real_target_passes(REAL_TLES, [30], 51.504646101082955,
                                -0.019568475850757397, target_azimuths=[az],
                                exclude_norad_ids=[67027], **kwargs)
    with pytest.raises(RuntimeError, match="No distinct real satellite"):
        make_real_target_passes(REAL_TLES, [30], 51.504646101082955,
                                -0.019568475850757397, target_azimuths=[(az + 180) % 360], **kwargs)


@pytest.mark.parametrize("azimuths", [[0, 90], [-1], [360]])
def test_invalid_azimuth_targets(azimuths):
    with pytest.raises(ValueError, match="one compass azimuth"):
        make_real_target_passes(REAL_TLES, [30], 51.5, -0.02, target_azimuths=azimuths)


def test_selection_skips_pass_crossing_search_boundary():
    passes = make_real_target_passes(REAL_TLES, [30], 51.504646101082955,
                                    -0.019568475850757397,
                                    epoch_utc="2026-08-24T05:29:53.573427Z",
                                    search_hours=1.2, tolerance_deg=40)
    # The 30-degree satellite is still rising at the end of this window;
    # choose the complete 60-degree satellite rather than a partial pass.
    assert passes[0]["metadata"]["norad_id"] == 47355
    assert passes[0]["metadata"]["set_epoch_offset_s"] <= 1.2 * 3600


def test_runner_refuses_old_synthetic_campaign(tmp_path):
    import argparse
    import json
    from Ray_Tracing.src.simulations.run_pass_snr_sweep import run
    (tmp_path / "campaign.json").write_text(json.dumps(dict(passes=[dict(
        orbit_source="synthetic SGP4 reference orbit")])))
    with pytest.raises(ValueError, match="real-TLE campaign"):
        run(argparse.Namespace(output=tmp_path))


def test_bins_edges_outages_and_empty_bins():
    bins = elevation_bin_statistics([0, 9.99, 10, 10, 90], [1, -np.inf, 0, 1, 5])
    assert bins[0]["sample_count"] == 2
    assert bins[0]["percent_above_threshold"] == 50
    assert bins[1]["percent_above_threshold"] == 50  # equality is excluded
    assert bins[-1]["percent_above_threshold"] == 100
    assert np.isnan(bins[2]["percent_above_threshold"])
    with pytest.raises(ValueError):
        elevation_bin_statistics([10], [np.nan])


def test_coherent_snr_and_noise_scaling():
    assert snr_db_from_cir([1e-7, -1e-7], 43, 10e6, 7) == -np.inf
    assert snr_db_from_cir([], 43, 10e6, 7) == -np.inf
    base = snr_db_from_cir([1e-7], 43, 10e6, 7)
    doubled = snr_db_from_cir([1e-7, 1e-7], 43, 10e6, 7)
    assert doubled - base == pytest.approx(20 * np.log10(2))
    assert snr_db_from_cir([1e-7], 43, 20e6, 7) == pytest.approx(base - 10 * np.log10(2))
    assert snr_db_from_cir([1e-7], 46, 10e6, 10) == pytest.approx(base)


def test_wideband_snr():
    nb = snr_db_from_cir([1e-7], 43, 10e6, 7)
    mean, eff = wideband_snr_db([1e-7], [1.5e-3], 43, 10e6, 7)
    assert mean == pytest.approx(nb) and eff == pytest.approx(nb)  # flat channel
    # Resolved paths add in power whatever their phase; antiphase is no longer a null.
    for a2 in (1e-7, -1e-7, 1e-7j):
        mean2, eff2 = wideband_snr_db([1e-7, a2], [0, 2e-6], 43, 10e6, 7)
        assert mean2 - nb == pytest.approx(10 * np.log10(2), abs=0.05)
        assert eff2 < mean2  # frequency selectivity costs capacity
    # Unresolved paths add coherently, as in the narrowband sum.
    close, _ = wideband_snr_db([1e-7, 1e-7], [0, 1e-10], 43, 10e6, 7)
    assert close == pytest.approx(snr_db_from_cir([1e-7, 1e-7], 43, 10e6, 7), abs=0.01)
    assert wideband_snr_db([1e-7], [0], 46, 10e6, 7)[0] == pytest.approx(nb + 3)
    assert wideband_snr_db([1e-7], [0], 43, 20e6, 7)[0] == pytest.approx(nb - 10 * np.log10(2))
    assert wideband_snr_db([], [], 43, 10e6, 7) == (-np.inf, -np.inf)


@pytest.mark.parametrize("target", [30, 60, 90])
def test_complete_target_pass(target):
    data = make_target_pass(target, 51.5047, -0.0195)
    meta, rows = data["metadata"], data["waypoints"]
    assert meta["actual_max_elevation_deg"] == pytest.approx(target, abs=1e-5)
    assert meta["inclination_deg"] == pytest.approx(53)
    times = np.array([w["t_s"] for w in rows])
    dt = meta["sample_interval_s"]
    assert np.diff(times) == pytest.approx(np.full(len(rows) - 1, dt))
    assert len(rows) * dt == pytest.approx(meta["duration_s"])
    assert times[0] == pytest.approx(dt / 2)
    assert times[-1] == pytest.approx(meta["duration_s"] - dt / 2)
    assert {w["leg"] for w in rows} == {"rising", "setting"}
    assert min(w["elev_deg"] for w in rows) >= 0
    assert max(w["elev_deg"] for w in rows) <= target + 1e-5


def test_saved_results_to_csv_and_plot(tmp_path):
    import csv
    import json
    import pickle
    from Ray_Tracing.src.simulations.run_pass_snr_sweep import collect

    ue = [1, 2, 1.5]
    cfg = {"TX_POWER_DBM": 43}
    meta = dict(directory="pass", target_max_elevation_deg=90,
                actual_max_elevation_deg=90, inclination_deg=53)
    manifest = dict(ue_scene_xyz_m=ue, bandwidth_hz=10e6,
                    noise_figure_db=7, temperature_k=290, passes=[meta])
    (tmp_path / "campaign.json").write_text(json.dumps(manifest))
    (tmp_path / "simulation_config.json").write_text(json.dumps(cfg))
    folder = tmp_path / "pass"
    folder.mkdir()
    waypoints = [dict(t_s=i * 5, elev_deg=el, az_deg=0, range_m=600e3, leg="rising")
                 for i, el in enumerate([10, 10, 20, 90])]
    with (folder / "waypoints.pkl").open("wb") as f:
        pickle.dump(dict(waypoints=waypoints), f)
    threshold = snr_db_from_cir([1e-7], 43, 10e6, 7)
    # A partial pass must never produce availability percentages.
    with pytest.raises(RuntimeError, match="Missing"):
        collect(tmp_path, threshold)
    for i, amps in enumerate([[1e-7], [2e-7], [], [2e-7]]):
        dest = folder / f"sample_{i:05d}"
        dest.mkdir()
        with (dest / "results_elev010deg.pkl").open("wb") as f:
            pickle.dump(dict(cfg_snapshot=cfg, nominal_rx_positions=[ue],
                             raw_results=[dict(amps=np.array(amps),
                                               taus=np.full(len(amps), 1e-3))]), f)
    collect(tmp_path, threshold)
    with (tmp_path / "elevation_bins.csv").open() as f:
        bins = list(csv.DictReader(f))
    assert float(bins[1]["percent_above_threshold"]) == 50
    assert float(bins[2]["percent_above_threshold"]) == 0
    assert int(bins[2]["sample_count"]) == 1  # No paths remains in the denominator.
    assert float(bins[8]["percent_above_threshold"]) == 100
    assert (tmp_path / "pass_snr.png").stat().st_size > 0
    with (tmp_path / "snr_samples.csv").open() as f:
        samples = list(csv.DictReader(f))
    assert float(samples[2]["snr_db"]) == -np.inf
    assert float(samples[1]["snr_db"]) == pytest.approx(float(samples[1]["snr_narrowband_db"]))
    collect(tmp_path, threshold, tx_power_dbm=53)
    with (tmp_path / "snr_samples.csv").open() as f:
        boosted = list(csv.DictReader(f))
    assert float(boosted[0]["snr_db"]) == pytest.approx(float(samples[0]["snr_db"]) + 10)
    assert float(boosted[0]["tx_power_dbm"]) == 53


@pytest.mark.parametrize("name", ["sc9", "sc6"])
def test_tr38821_link_budget_reproduced(name):
    """A free-space LoS path at 30 deg from 600 km gives TR 38.821 Table 6.1.3.3-1 CNR."""
    from Ray_Tracing.src.utils.pass_snr import (TR38821_CASES, band_gain, dish_gain_db,
                                               snr_from_band_gain, tr38811_losses_db)
    c = TR38821_CASES[name]
    R, h, el = 6371e3, 600e3, np.radians(30)
    d = np.sqrt((R + h) ** 2 - (R * np.cos(el)) ** 2) - R * np.sin(el)
    a = 299792458.0 / c["frequency_hz"] / (4 * np.pi * d)
    if c["rx_model"] == "mrc":   # CP wave on +-45 deg ports: half power each
        ports = [a / np.sqrt(2), 1j * a / np.sqrt(2)]
    else:                        # dish pointed at the satellite
        ports = [a * 10 ** (dish_gain_db(0.0, c["frequency_hz"], c["dish_diameter_m"],
                                         c["dish_gain_dbi"]) / 20)]
    gain = sum(band_gain([p], [1e-3], c["bandwidth_hz"], c["subcarrier_spacing_hz"]) for p in ports)
    eirp = c["eirp_dbw_per_mhz"] + 30 + 10 * np.log10(c["bandwidth_hz"] / 1e6)
    loss = tr38811_losses_db(c["frequency_hz"], 30) + c["shadow_margin_db"]
    cnr = snr_from_band_gain(gain, eirp - loss, c["bandwidth_hz"], c["noise_figure_db"],
                             c["antenna_temperature_k"])[0]
    assert cnr == pytest.approx(c["tr_cnr_db"], abs=0.15)


def test_dish_pattern_edges():
    from Ray_Tracing.src.utils.pass_snr import dish_gain_db
    g = dish_gain_db(np.radians([0, 0.735, 120]), 20e9, 0.6, 39.7)
    assert g[0] == pytest.approx(39.7)
    assert g[1] == pytest.approx(39.7 - 3, abs=0.1)   # half-power beamwidth ~1.47 deg
    assert g[2] == -np.inf                             # nothing from behind the dish


def test_target_pass_heading_and_side():
    east = make_target_pass(60, 51.5047, -0.0195, 600, 97.8, 10, northbound=False, side=1)["metadata"]
    west = make_target_pass(60, 51.5047, -0.0195, 600, 97.8, 10, northbound=False, side=-1)["metadata"]
    assert not east["northbound"] and not west["northbound"]
    assert east["actual_max_elevation_deg"] == pytest.approx(60, abs=1e-3)
    assert west["actual_max_elevation_deg"] == pytest.approx(60, abs=1e-3)
    assert 0 < east["peak_azimuth_deg"] < 180 < west["peak_azimuth_deg"]


def test_diffuse_paths_add_power_not_amplitude():
    """N in-phase diffuse paths splitting one path's power must not gain N (Sionna phases)."""
    from Ray_Tracing.src.utils.pass_snr import band_gain
    n, a = 1000, 1e-8
    amps = np.r_[a, np.full(n, a / np.sqrt(n))]
    taus = np.r_[1e-3, np.full(n, 1e-3 + 1e-9)]
    diffuse = np.r_[False, np.ones(n, bool)]
    g = band_gain(amps, taus, 30e6, incoherent=diffuse)
    assert g.mean() == pytest.approx(2 * a ** 2, rel=1e-6)
    assert band_gain(amps, taus, 30e6).mean() > 500 * a ** 2   # the coherent artifact
