#!/usr/bin/env python3
"""Convert a pass pickle into a dense ns-3 WaypointMobilityModel CSV.
"""

import argparse
import csv
import pickle
import sys

import numpy as np

from Ray_Tracing.src.utils.channel_utils import tx_position_from_pass_sample, tx_velocity_scene_frame


def _row_to_waypoint(row: dict) -> dict:
    """
    Adapt one dense `sample_pass_table` row (which has `range_km`, not
    `range_m`) into the waypoint dict shape `tx_position_from_pass_sample`
    expects (elev_deg, az_deg, range_m, enu_vel_kms).
    """
    return {
        "elev_deg": row["elev_deg"],
        "az_deg": row["az_deg"],
        "range_m": row["range_km"] * 1e3,
        "enu_vel_kms": row["enu_vel_kms"],
    }


def export(pkl_path: str, output_path: str, scene_center_xy, scene_radius: float,
           scene_east_heading_deg: float) -> int:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    rows = data["pass"]["rows"]
    if not rows:
        raise ValueError(f"{pkl_path}: pass['rows'] is empty -- nothing to export")

    t0 = rows[0]["t_s"]
    out_rows = []
    for row in rows:
        wp = _row_to_waypoint(row)
        tx_pos, _illum_angle = tx_position_from_pass_sample(
            wp, scene_center_xy, scene_radius, scene_east_heading_deg=scene_east_heading_deg,
        )
        v_scene = tx_velocity_scene_frame(wp, scene_east_heading_deg=scene_east_heading_deg)
        out_rows.append({
            "t_s": row["t_s"] - t0,
            "x": tx_pos[0], "y": tx_pos[1], "z": tx_pos[2],
            "vx": float(v_scene[0]), "vy": float(v_scene[1]), "vz": float(v_scene[2]),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t_s", "x", "y", "z", "vx", "vy", "vz"])
        writer.writeheader()
        writer.writerows(out_rows)

    step_s = out_rows[1]["t_s"] - out_rows[0]["t_s"] if len(out_rows) > 1 else float("nan")
    duration_s = out_rows[-1]["t_s"] - out_rows[0]["t_s"]
    max_speed = max(float(np.linalg.norm([r["vx"], r["vy"], r["vz"]])) for r in out_rows)
    print(f"[export_ns3_waypoints] Wrote {len(out_rows)} rows -> {output_path}")
    print(f"  step_s~={step_s:.3f}  duration_s={duration_s:.1f}  "
          f"max |v_scene|={max_speed:.1f} m/s")
    print(f"  scene_center={tuple(scene_center_xy)}  scene_radius={scene_radius:.2f} m  "
          f"east_heading={scene_east_heading_deg} deg")
    return len(out_rows)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="Path to a generate_pass_waypoints.py output pickle")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--scene-center", type=float, nargs=2, default=[0.0, 0.0],
                    metavar=("X", "Y"), help="Scene center [x, y] in the ground plane [m]")
    p.add_argument("--scene-radius", type=float, required=True,
                    help="Scene circumscribing radius [m] (SCENE_RADIUS in sim_config.json)")
    p.add_argument("--scene-east-heading", type=float, default=90.0,
                    help="Compass heading of the scene's +X axis [deg] "
                         "(SCENE_EAST_HEADING_DEG in sim_config.json)")
    return p


def main():
    args = build_arg_parser().parse_args()
    n = export(args.pkl, args.output, args.scene_center, args.scene_radius, args.scene_east_heading)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
