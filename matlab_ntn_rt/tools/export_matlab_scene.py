"""Merge a Mitsuba/Sionna PLY scene into the single STL that MATLAB raytrace requires.

MATLAB's Cartesian raytrace takes one triangulation and one material
(rfprop.RayTracing SurfaceMaterial is a 1x1 string), so the per-shape material
assignment has to leave the geometry and travel alongside it: the companion .mat
carries a per-triangle material index that +ntnrt/+mat/perInteractionMaterials.m
uses to recover the real material at each interaction point after the solve.

Usage: python export_matlab_scene.py sieg wharf_du
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.io import savemat

from Ray_Tracing.matlab_ntn_rt.tools.ply_stl import read_ply, write_stl

OUT = Path(__file__).resolve().parent.parent / "data"

SCENES = {
    "sieg": {
        "xml": "/path/to/sieg_scene/sieg_notrees.xml",
        "ground_xml": "/path/to/sieg_scene/sieg_ground_only.xml",
    },
    "wharf_du": {
        "xml": "/path/to/wharf_scene_v3/du/canarywharf3_du.xml",
        "ground_xml": None,
    },
}


def parse_scene(xml_path):
    """Return [(shape_id, ply_path, material_name)] in document order."""
    root = ET.parse(xml_path).getroot()
    base = Path(xml_path).parent
    shapes = []
    for sh in root.findall("shape"):
        fn = sh.find("string[@name='filename']")
        ref = sh.find("ref[@name='bsdf']")
        if fn is None:
            continue
        if sh.find("transform") is not None:
            raise ValueError(f"{sh.get('id')}: shape transforms are not handled")
        # The XML ref is authoritative: Terrain -> itu_medium_dry_ground, whereas
        # the filename suffix would say "Terrain".
        mat = ref.get("id").removeprefix("mat-") if ref is not None else "unknown"
        shapes.append((sh.get("id"), base / fn.get("value"), mat))
    return shapes


def merge(shapes):
    """Concatenate shapes into one mesh, tracking per-triangle material and shape."""
    V, F, mat_idx, shape_idx, names = [], [], [], [], []
    nv = 0
    for si, (_, ply, mat) in enumerate(shapes):
        v, f = read_ply(ply)
        if mat not in names:
            names.append(mat)
        V.append(v)
        F.append(f + nv)
        mat_idx.append(np.full(len(f), names.index(mat), np.int32))
        shape_idx.append(np.full(len(f), si, np.int32))
        nv += len(v)
    return (np.concatenate(V), np.concatenate(F), np.concatenate(mat_idx),
            np.concatenate(shape_idx), names)


def export(tag, spec):
    shapes = parse_scene(spec["xml"])
    V, F, mat_idx, shape_idx, names = merge(shapes)
    write_stl(OUT / f"{tag}.stl", V, F)

    centroid = V[F].mean(axis=1)
    lo, hi = V.min(0), V.max(0)
    ctr = 0.5 * (lo + hi)
    meta = {
        "scene": tag,
        "source_xml": spec["xml"],
        "n_shapes": len(shapes),
        "n_vertices": int(len(V)),
        "n_triangles": int(len(F)),
        "materials": names,
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "center_xy": ctr[:2].tolist(),
        # Circumradius of the XY footprint about its bbox centre; compare against
        # SCENE_RADIUS in sim_config.json.
        "radius_xy": float(np.linalg.norm((V[:, :2] - ctr[:2]), axis=1).max()),
    }

    savemat(OUT / f"{tag}_materials.mat", {
        "faceMaterial": (mat_idx + 1).reshape(-1, 1),     # 1-based for MATLAB
        "faceShape": (shape_idx + 1).reshape(-1, 1),
        "materialNames": np.array(names, dtype=object).reshape(-1, 1),
        "shapeNames": np.array([s[0] for s in shapes], dtype=object).reshape(-1, 1),
        "faceCentroid": centroid,
        "sceneTag": tag,
        "nTriangles": len(F),
    }, do_compression=True)

    if spec["ground_xml"]:
        gV, gF, *_ = merge(parse_scene(spec["ground_xml"]))
        write_stl(OUT / f"{tag}_ground_only.stl", gV, gF)
        meta["ground_triangles"] = int(len(gF))

    (OUT / f"{tag}_scene_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{tag}: {len(shapes)} shapes, {len(F)} tris, materials={names}")
    print(f"  centre_xy={meta['center_xy']} radius_xy={meta['radius_xy']:.1f} m"
          f" z=[{lo[2]:.1f},{hi[2]:.1f}]")
    if "ground_triangles" in meta:
        print(f"  ground-only: {meta['ground_triangles']} tris")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in (sys.argv[1:] or list(SCENES)):
        export(tag, SCENES[tag])
