"""Emit a Sionna-loadable copy of the UW campus (Sieg) scene, plus a bare-earth
companion for the terrain-following occupancy grid. Originals are untouched.

The authored sieg.xml does not load at all: itu_tree is not in Sionna's ITU
table, and Sionna instantiates every top-level <bsdf> whether a shape uses it
or not, so the material block itself must go -- not just the tree shapes.

Removing vegetation drops 30 shapes (trunk + canopy per tree) and makes the
scene optimistic in LoS and K. See docs/pipelines.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

# Scene assets live outside the repo. Override with $SIEG_SCENE_DIR; the default
# is a sieg_scene/ directory sitting alongside the repo.
SCENE_DIR = os.environ.get(
    "SIEG_SCENE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sieg_scene"),
)
DEFAULT_SRC = os.path.join(SCENE_DIR, "sieg.xml")

# Shapes whose id matches this are vegetation (canopies AND trunks) -> dropped.
TREE_SHAPE_RE = r"^mesh-Cylinder"

# Materials that exist only to support vegetation -> dropped once unreferenced.
DROP_MATERIALS = ("mat-itu_tree", "mat-itu_wood")

# old bsdf id -> new bsdf id. Empty: the ground materials keep their true ITU
# identities, made loadable above 10 GHz by itu_range_patch.widen(). Callers can
# still pass their own mapping to sanitize_scene().
MATERIAL_RENAME: dict[str, str] = {}

# Shapes that make up the bare-earth scene used for terrain-relief ray casting.
GROUND_SHAPE_IDS = ("mesh-Terrain", "mesh-None_water")

# ITU-R P.2040-1 coefficients for the two ground materials.
# Feed these to cfg["MATERIAL_PARAMS"]; every entry MUST set
# scattering_coefficient explicitly, because build_scene_batch defaults it to 0
# (i.e. omitting it silently disables diffuse scattering for that surface).
GROUND_MATERIAL_PARAMS = {
    "itu_medium_dry_ground": {"a": 15.0, "b": -0.1, "c": 0.035, "d": 1.63,
                              "scattering_coefficient": 0.3, "xpd_coefficient": 0.0},
    "itu_wet_ground":        {"a": 30.0, "b": -0.4, "c": 0.15,  "d": 1.30,
                              "scattering_coefficient": 0.0, "xpd_coefficient": 0.0},
}


def _indent(elem, level=0):
    """Minimal pretty-printer (ET.indent equivalent, kept dependency-free)."""
    pad = "\n" + "\t" * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "\t"
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def sanitize_scene(src_xml: str = DEFAULT_SRC,
                   out_dir: str | None = None,
                   scene_name: str = "sieg_notrees",
                   ground_name: str = "sieg_ground_only",
                   tree_shape_re: str = TREE_SHAPE_RE,
                   drop_materials: tuple = DROP_MATERIALS,
                   material_rename: dict | None = None,
                   ground_shape_ids: tuple = GROUND_SHAPE_IDS) -> dict:
    """
    Write a vegetation-free, frequency-valid scene plus a ground-only scene.

    Returns a report dict (also written as ``<scene_name>_meta.json``) with the
    shape/material census and the applied material remapping, so a run can
    always be traced back to exactly which geometry it used.
    """
    material_rename = dict(MATERIAL_RENAME if material_rename is None
                           else material_rename)
    out_dir = out_dir or os.path.dirname(os.path.abspath(src_xml))
    os.makedirs(out_dir, exist_ok=True)

    tree = ET.parse(src_xml)
    root = tree.getroot()

    shapes_in = root.findall("shape")
    bsdfs_in = root.findall("bsdf")

    # ---- 1. drop vegetation shapes -----------------------------------------
    tree_re = re.compile(tree_shape_re)
    dropped_shapes = []
    for sh in shapes_in:
        if tree_re.match(sh.get("id") or ""):
            root.remove(sh)
            dropped_shapes.append(sh.get("id"))

    # ---- 2. drop the now-unreferenced vegetation materials ------------------
    # Sionna instantiates every top-level <bsdf> regardless of references, so
    # leaving mat-itu_tree in place would still raise.
    dropped_materials = []
    for bs in bsdfs_in:
        if (bs.get("id") or "") in drop_materials:
            root.remove(bs)
            dropped_materials.append(bs.get("id"))

    # ---- 3. rename the ground materials to frequency-valid identities -------
    for bs in root.findall("bsdf"):
        old = bs.get("id") or ""
        if old in material_rename:
            new = material_rename[old]
            bs.set("id", new)
            if bs.get("name") is not None:
                bs.set("name", new)
    for ref in root.iter("ref"):
        old = ref.get("id") or ""
        if old in material_rename:
            ref.set("id", material_rename[old])

    _indent(root)
    out_xml = os.path.join(out_dir, f"{scene_name}.xml")
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)

    # ---- 4. bare-earth companion scene --------------------------------------
    # Same ray set is cast into this and into the full scene; the *difference*
    # in hit height is what separates buildings from terrain relief, which an
    # absolute height threshold cannot do (building and terrain z-ranges
    # overlap in this scene).
    g_tree = ET.parse(out_xml)
    g_root = g_tree.getroot()
    keep_mats = set()
    for sh in list(g_root.findall("shape")):
        if (sh.get("id") or "") in ground_shape_ids:
            for ref in sh.iter("ref"):
                keep_mats.add(ref.get("id"))
        else:
            g_root.remove(sh)
    for bs in list(g_root.findall("bsdf")):
        if (bs.get("id") or "") not in keep_mats:
            g_root.remove(bs)

    _indent(g_root)
    out_ground = os.path.join(out_dir, f"{ground_name}.xml")
    g_tree.write(out_ground, encoding="utf-8", xml_declaration=False)

    # ---- 5. provenance -------------------------------------------------------
    final_shapes = [s.get("id") for s in root.findall("shape")]
    final_mats = [b.get("id") for b in root.findall("bsdf")]
    report = {
        "src_xml": os.path.abspath(src_xml),
        "out_xml": os.path.abspath(out_xml),
        "out_ground_xml": os.path.abspath(out_ground),
        "n_shapes_in": len(shapes_in),
        "n_shapes_out": len(final_shapes),
        "n_dropped_shapes": len(dropped_shapes),
        "dropped_shapes": dropped_shapes,
        "dropped_materials": dropped_materials,
        "material_rename": material_rename,
        "materials_out": sorted(final_mats),
        "shapes_out": final_shapes,
        "ground_shapes": list(ground_shape_ids),
        "ground_material_params": GROUND_MATERIAL_PARAMS,
    }
    with open(os.path.join(out_dir, f"{scene_name}_meta.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[sanitize] {len(shapes_in)} -> {len(final_shapes)} shapes "
          f"({len(dropped_shapes)} vegetation shapes dropped)")
    print(f"[sanitize] dropped materials : {dropped_materials}")
    print(f"[sanitize] renamed materials : {material_rename}")
    print(f"[sanitize] materials out     : {sorted(final_mats)}")
    print(f"[sanitize] wrote {out_xml}")
    print(f"[sanitize] wrote {out_ground}")
    return report


def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out_dir", default=None,
                   help="defaults to the source XML's directory (needs meshes/)")
    p.add_argument("--scene_name", default="sieg_notrees")
    p.add_argument("--ground_name", default="sieg_ground_only")
    a = p.parse_args()
    sanitize_scene(a.src, a.out_dir, a.scene_name, a.ground_name)


if __name__ == "__main__":
    _cli()
