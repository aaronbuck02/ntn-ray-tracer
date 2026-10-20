"""Check the max-z rasterisation against Mitsuba's raycast occupancy grid.

+ntnrt/+geom/occupancyGrid.m replaces the Python pipeline's Mitsuba downward
ray cast with a rasterisation, because MATLAB has no scene ray caster exposed.
This runs the identical rasterisation in Python and diffs the resulting
free/occupied sets against the grid_meta stored in rx_positions_sieg.pkl, so the
algorithm is validated independently of whether MATLAB can be launched.

Usage: python validate_grid_algorithm.py [sieg]
"""

import pickle
import sys
from pathlib import Path

import numpy as np

from Ray_Tracing.matlab_ntn_rt.tools.export_matlab_scene import SCENES, merge, parse_scene

SRC = Path("/path/to/rician_urban_sim")


def max_z(V, F, xc, yc):
    """Per-cell max surface height; NaN where no facet covers the cell centre."""
    A, B, D = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    z = np.full((len(xc), len(yc)), np.nan)
    res = xc[1] - xc[0]

    lo = np.minimum(np.minimum(A, B), D)
    hi = np.maximum(np.maximum(A, B), D)
    i0 = np.maximum(0, np.floor((lo[:, 0] - xc[0]) / res).astype(int))
    i1 = np.minimum(len(xc) - 1, np.ceil((hi[:, 0] - xc[0]) / res).astype(int))
    j0 = np.maximum(0, np.floor((lo[:, 1] - yc[0]) / res).astype(int))
    j1 = np.minimum(len(yc) - 1, np.ceil((hi[:, 1] - yc[0]) / res).astype(int))

    den = (B[:, 1] - D[:, 1]) * (A[:, 0] - D[:, 0]) + (D[:, 0] - B[:, 0]) * (A[:, 1] - D[:, 1])

    for f in range(len(F)):
        if abs(den[f]) < 1e-12 or i1[f] < i0[f] or j1[f] < j0[f]:
            continue
        ii, jj = np.meshgrid(np.arange(i0[f], i1[f] + 1), np.arange(j0[f], j1[f] + 1), indexing="ij")
        px, py = xc[ii.ravel()], yc[jj.ravel()]
        l1 = ((B[f, 1] - D[f, 1]) * (px - D[f, 0]) + (D[f, 0] - B[f, 0]) * (py - D[f, 1])) / den[f]
        l2 = ((D[f, 1] - A[f, 1]) * (px - D[f, 0]) + (A[f, 0] - D[f, 0]) * (py - D[f, 1])) / den[f]
        l3 = 1 - l1 - l2
        in_tri = (l1 >= -1e-12) & (l2 >= -1e-12) & (l3 >= -1e-12)
        if not in_tri.any():
            continue
        zf = l1[in_tri] * A[f, 2] + l2[in_tri] * B[f, 2] + l3[in_tri] * D[f, 2]
        k = (ii.ravel()[in_tri], jj.ravel()[in_tri])
        cur = z[k]
        z[k] = np.where(np.isnan(cur), zf, np.maximum(cur, zf))
    return z


def main(tag="sieg"):
    spec = SCENES[tag]
    V, F, *_ = merge(parse_scene(spec["xml"]))
    gV, gF, *_ = merge(parse_scene(spec["ground_xml"]))

    ref = pickle.load(open(SRC / f"rx_positions_{tag}.pkl", "rb"))["meta"]["grid_meta"]
    res, margin, thresh = ref["grid_res"], ref["margin"], 1.5

    lo, hi = V[:, :2].min(0), V[:, :2].max(0)
    xc = lo[0] + np.arange(int(np.ceil((hi[0] + res - lo[0]) / res))) * res
    yc = lo[1] + np.arange(int(np.ceil((hi[1] + res - lo[1]) / res))) * res

    top, gnd = max_z(V, F, xc, yc), max_z(gV, gF, xc, yc)
    both = np.isfinite(top) & np.isfinite(gnd)
    ground_hit = both & ((top - gnd) <= thresh)

    in_valid = np.zeros_like(both)
    in_valid[np.ix_((xc >= lo[0] + margin) & (xc <= hi[0] - margin),
                    (yc >= lo[1] + margin) & (yc <= hi[1] - margin))] = True

    free_valid = ground_hit & in_valid
    occupied = both & ~ground_hit

    print(f"{tag}: grid {len(xc)}x{len(yc)} @ {res} m")
    print(f"{'':14}{'raster':>10}{'mitsuba':>10}{'delta':>8}")
    for name, got, want in [("n_total", free_valid.size, ref["n_total"]),
                            ("n_free", int(free_valid.sum()), ref["n_free"]),
                            ("n_free_full", int(ground_hit.sum()), ref["n_free_full"]),
                            ("n_occupied", int(occupied.sum()), ref["n_occupied"])]:
        print(f"  {name:12}{got:>10}{want:>10}{got-want:>8}")

    # Set-level agreement on occupied cells, which is the grid builder's own output.
    oi, oj = np.nonzero(occupied)
    got = {(round(float(xc[i]), 4), round(float(yc[j]), 4)) for i, j in zip(oi, oj)}
    want = {(round(float(x), 4), round(float(y), 4)) for x, y in ref["occupied_xy"]}
    inter = got & want
    print(f"  occupied set: {len(inter)} shared, {len(got-want)} raster-only, "
          f"{len(want-got)} mitsuba-only  (Jaccard {len(inter)/len(got|want):.4f})")

    gz = gnd[np.isfinite(gnd)]
    print(f"  ground_z range raster=({gz.min():.4f},{gz.max():.4f}) "
          f"mitsuba=({ref['ground_z_range'][0]:.4f},{ref['ground_z_range'][1]:.4f})")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["sieg"]))
