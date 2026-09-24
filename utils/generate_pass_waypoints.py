#!/usr/bin/env python3
"""CLI to generate orbit-pass waypoints for the runners' --waypoints_file mode.

Wraps orbital_utils.find_pass / generate_waypoints. See docs/physics.md.
"""

import argparse
import pickle
import sys
from pathlib import Path
import os

import numpy as np

# Import root is the directory containing Ray_Tracing/, so the
# Ray_Tracing.src.* package paths resolve however this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Ray_Tracing.src.utils.orbital_utils import (
    make_reference_satrec,
    REFERENCE_LEO_ALTITUDES_KM,
    orbital_period_s,
    find_pass,
    find_pass_with_geometry_class,
    PASS_GEOMETRY_BANDS,
    generate_waypoints,
    doppler_window_validity,
    parse_tle_catalog,
    search_tle_catalog_for_pass,
    best_pass_in_catalog,
)
from sgp4.api import jday


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate SGP4 pass waypoints for the NTN RT pipeline")
    p.add_argument("--reference", type=str, default="leo600",
                    choices=["leo600", "leo1200"],
                    help="3GPP TR 38.821 reference LEO altitude (ignored if --tle_catalog given)")
    p.add_argument("--tle_catalog", type=str, default=None,
                    help="Path to a bulk 3-line-format TLE text file covering many real "
                         "satellites (e.g. celestrak's gp.php?GROUP=starlink&FORMAT=TLE). "
                         "When given, searches across all of them for a real pass instead "
                         "of using a synthetic orbit.")
    p.add_argument("--max_satellites", type=int, default=None,
                    help="Cap how many catalog entries to propagate (--tle_catalog mode "
                         "only); bulk catalogs can have thousands of satellites. "
                         "None = scan the whole catalog.")
    p.add_argument("--catalog_select", type=str, default="soonest",
                    choices=["soonest", "highest_elev"],
                    help="--tle_catalog mode only: how to pick one satellite among all "
                         "real ones with a qualifying pass. 'soonest' = earliest real pass "
                         "in the search window, 'highest_elev' = real pass with the "
                         "largest max elevation.")
    p.add_argument("--gs_lat", type=float, required=True, help="Ground station latitude [deg]")
    p.add_argument("--gs_lon", type=float, required=True, help="Ground station longitude [deg]")
    p.add_argument("--gs_alt_km", type=float, default=0.0, help="Ground station altitude [km]")
    p.add_argument("--inclination", type=float, default=53.0,
                    help="Orbit inclination [deg] (synthetic-orbit mode only)")
    p.add_argument("--raan", type=float, default=0.0,
                    help="RAAN [deg] (synthetic-orbit mode only)")
    p.add_argument("--arg_lat", type=float, default=0.0,
                    help="Argument of latitude (mean anomaly) at epoch [deg] "
                         "(synthetic-orbit mode only) — controls where in the "
                         "orbit the satellite starts; sweep this if --search_hours "
                         "doesn't find a pass")
    p.add_argument("--epoch", type=str, default=None,
                    help="ISO8601 UTC epoch to start the pass search from. "
                         "Defaults to 2026-01-01T00:00:00 in both modes.")
    p.add_argument("--min_elev", type=float, default=10.0, help="Mask elevation [deg]")
    p.add_argument("--elev_step", type=float, default=10.0, help="Waypoint spacing [deg]")
    p.add_argument("--leg", type=str, default="rising", choices=["rising", "setting"])
    p.add_argument("--search_hours", type=float, default=6.0,
                    help="How long to scan for a pass, starting at --epoch "
                         "(ignored when --pass_geometry is not 'any' in synthetic-orbit mode)")
    p.add_argument("--pass_geometry", type=str, default="any",
                    choices=["any"] + list(PASS_GEOMETRY_BANDS),
                    help="Synthetic-orbit mode: instead of taking whatever pass the fixed "
                         "--raan/--arg_lat produces, search a grid of RAAN/arg_lat "
                         "combinations for an invented orbit whose max elevation matches "
                         "this class. --tle_catalog mode: filter real satellites' real "
                         "passes to this class instead of inventing anything -- 'overhead' "
                         "= near-zenith, 'near' = close but not overhead, 'far' = still "
                         "above --min_elev but low on the horizon (long slant range).")
    p.add_argument("--geometry_raan_grid", type=int, default=8,
                    help="RAAN grid resolution for --pass_geometry search")
    p.add_argument("--geometry_arg_lat_grid", type=int, default=8,
                    help="Argument-of-latitude grid resolution for --pass_geometry search")
    p.add_argument("--step_s", type=float, default=5.0, help="Pass-table sampling interval [s]")
    p.add_argument("--doppler_window_s", type=float, default=0.01,
                    help="Candidate snapshot duration for the Doppler quasi-static check")
    p.add_argument("--max_drift_deg", type=float, default=0.05,
                    help="Max allowed elevation drift within doppler_window_s")
    p.add_argument("--output", type=str, default="pass_waypoints.pkl")
    return p


def _parse_iso_epoch(epoch_str):
    year, rest = epoch_str.split("-", 1)
    month, rest = rest.split("-", 1)
    day, rest = rest.split("T", 1)
    hour, minute, sec = rest.split(":")
    return int(year), int(month), int(day), int(hour), int(minute), float(sec)


def main():
    args = build_arg_parser().parse_args()

    using_catalog = bool(args.tle_catalog)

    if using_catalog:
        with open(args.tle_catalog, "r") as f:
            catalog_text = f.read()
        catalog = parse_tle_catalog(catalog_text)
        print(f"Loaded catalog: {len(catalog)} real satellites from {args.tle_catalog}")

        epoch_str = args.epoch or "2026-01-01T00:00:00"
        year, month, day, hour, minute, sec = _parse_iso_epoch(epoch_str)
        jd_start, fr_start = jday(year, month, day, hour, minute, sec)

        geometry_filter = None if args.pass_geometry == "any" else args.pass_geometry
        n_scan = args.max_satellites or len(catalog)
        print(f"Searching {n_scan} of {len(catalog)} real satellites for a pass above "
              f"{args.min_elev} deg within {args.search_hours} h of {epoch_str}"
              + (f", filtered to '{args.pass_geometry}' geometry" if geometry_filter else "")
              + " ...")

        matches = search_tle_catalog_for_pass(
            catalog, args.gs_lat, args.gs_lon, args.gs_alt_km,
            jd_start, fr_start, search_duration_s=args.search_hours * 3600.0,
            min_elev_deg=args.min_elev, step_s=args.step_s,
            geometry_class=geometry_filter, max_satellites=args.max_satellites,
        )
        if not matches:
            print(f"No real satellite in the catalog has a qualifying pass. Try a "
                  f"longer --search_hours, a lower --min_elev, dropping --pass_geometry, "
                  f"or a bigger/different --tle_catalog.")
            sys.exit(1)

        print(f"Found {len(matches)} real satellite(s) with a qualifying pass:")
        for m in sorted(matches, key=lambda m: m["pass"]["rows"][0]["t_s"])[:10]:
            print(f"  {m['name']:<25s} NORAD {m['satnum']:<6d} "
                  f"max_elev={m['max_elev_deg']:6.2f} deg  class={m['geometry_class']}")
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")

        chosen = best_pass_in_catalog(matches, select=args.catalog_select)
        satrec, pass_dict = chosen["satrec"], chosen["pass"]
        print(f"\nSelected ({args.catalog_select}): {chosen['name']}  "
              f"(NORAD {chosen['satnum']}, max elevation "
              f"{chosen['max_elev_deg']:.2f} deg, class={chosen['geometry_class']})")

    else:
        epoch_str = args.epoch or "2026-01-01T00:00:00"
        year, month, day, hour, minute, sec = _parse_iso_epoch(epoch_str)
        jd_start, fr_start = jday(year, month, day, hour, minute, sec)
        epoch_kwargs = dict(epoch_year=year, epoch_month=month, epoch_day=day,
                             epoch_hour=hour, epoch_min=minute, epoch_sec=sec)
        altitude_km = REFERENCE_LEO_ALTITUDES_KM[args.reference]

        if args.pass_geometry == "any":
            satrec = make_reference_satrec(
                altitude_km=altitude_km, inclination_deg=args.inclination,
                raan_deg=args.raan, arg_lat_deg=args.arg_lat, **epoch_kwargs,
            )
            period_s = orbital_period_s(altitude_km)
            print(f"Synthetic reference orbit: {args.reference}  period={period_s/60:.2f} min")

            pass_dict = find_pass(
                satrec, args.gs_lat, args.gs_lon, args.gs_alt_km,
                jd_start, fr_start, search_duration_s=args.search_hours * 3600.0,
                min_elev_deg=args.min_elev, step_s=args.step_s,
            )
            if pass_dict is None:
                print(f"No pass found above {args.min_elev} deg within {args.search_hours} h "
                      f"of epoch {epoch_str}. Try a different --arg_lat, --inclination, "
                      f"a longer --search_hours, or use --pass_geometry to search "
                      f"automatically over RAAN/arg_lat.")
                sys.exit(1)
            print(f"Pass found: {len(pass_dict['rows'])} samples, "
                  f"max elevation = {pass_dict['max_elev_deg']:.2f} deg")

        else:
            # The actual RAAN/arg_lat search lives in orbital_utils —
            # this script just supplies the satellite/altitude/epoch and
            # reports the result.
            def make_satrec_fn(raan_deg, arg_lat_deg):
                return make_reference_satrec(
                    altitude_km=altitude_km, inclination_deg=args.inclination,
                    raan_deg=raan_deg, arg_lat_deg=arg_lat_deg, **epoch_kwargs,
                )

            print(f"Synthetic reference orbit: {args.reference}  "
                  f"searching for a '{args.pass_geometry}' pass "
                  f"({args.geometry_raan_grid}x{args.geometry_arg_lat_grid} grid)...")
            result = find_pass_with_geometry_class(
                make_satrec_fn, args.gs_lat, args.gs_lon, args.gs_alt_km,
                jd_start, fr_start, min_elev_deg=args.min_elev,
                geometry_class=args.pass_geometry, step_s=args.step_s,
                raan_grid=args.geometry_raan_grid, arg_lat_grid=args.geometry_arg_lat_grid,
            )
            if result is None:
                print(f"No pass found above {args.min_elev} deg anywhere in the "
                      f"RAAN/arg_lat grid. Try a finer grid (--geometry_raan_grid / "
                      f"--geometry_arg_lat_grid) or a lower --min_elev.")
                sys.exit(1)

            satrec, pass_dict = result["satrec"], result["pass"]
            lo, hi = result["band"]
            match_note = "matched" if result["matched_band"] else "CLOSEST AVAILABLE (no exact match in grid)"
            print(f"Pass found ({match_note}): {len(pass_dict['rows'])} samples, "
                  f"max elevation = {pass_dict['max_elev_deg']:.2f} deg "
                  f"(target band [{lo:.0f}, {hi:.0f}] deg for '{args.pass_geometry}')")
            print(f"  RAAN={result['raan_deg']:.1f} deg  arg_lat={result['arg_lat_deg']:.1f} deg "
                  f"(chosen by the geometry search, not --raan/--arg_lat)")

    waypoints = generate_waypoints(
        pass_dict, satrec, args.gs_lat, args.gs_lon, args.gs_alt_km,
        elev_step_deg=args.elev_step, mask_elev_deg=args.min_elev, leg=args.leg,
    )

    validity = [
        doppler_window_validity(pass_dict, wp, args.doppler_window_s, args.max_drift_deg)
        for wp in waypoints
    ]

    print(f"\n{'elev_target':>12} {'matched':>9} {'az_deg':>8} {'range_km':>10} "
          f"{'|v|_km/s':>9} {'window_ok':>10}")
    for wp, v in zip(waypoints, validity):
        speed = float(np.linalg.norm(wp["enu_vel_kms"]))
        print(f"{wp['elev_deg']:12.1f} {wp['matched_elev_deg']:9.2f} "
              f"{wp['az_deg']:8.2f} {wp['range_km']:10.1f} {speed:9.3f} "
              f"{'yes' if v['valid'] else 'NO':>10}")

    out = {
        "pass": pass_dict,
        "waypoints": waypoints,
        "window_validity": validity,
        "config": vars(args),
    }
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved {len(waypoints)} waypoints to {args.output}")


if __name__ == "__main__":
    main()
