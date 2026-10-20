"""Rebuild the Canary Wharf scene with one ITU material per physical role.

The authored scene resolves every surface to concrete or metal, so wall, roof
and ground are indistinguishable to the solver. Geometry, not the material
name, determines the role here: the "concrete" meshes are vertical facades and
the "metal" meshes are horizontal roof slabs.

Also caps the open shells, testing each candidate cap against existing geometry
so overlap seams are not mistaken for holes. See docs/pipelines.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import xml.etree.ElementTree as ET

import numpy as np

from Ray_Tracing.src.utils.scene_gen_ppp import write_ply

# Scene assets live outside the repo. Override with $WHARF_SCENE_DIR.
SCENE_DIR = os.environ.get("WHARF_SCENE_DIR", "/path/to/wharf_scene_v3")
DEFAULT_SRC = os.path.join(SCENE_DIR, "canarywharf3.xml")
DEFAULT_OUT = os.path.join(SCENE_DIR, "du")

# Source bsdf id -> role. Names are the exporter's, not the physics; see above.
SRC_MATERIAL_ROLE = {
    "mat-itu_concrete": "wall",
    "mat-itu_metal":    "roof",
}

# The single ground shape, identified by its mesh: it references the building
# material in the source, so the material ref cannot separate it from the walls.
GROUND_MESH = "Plane.ply"

# role -> ITU material name (the id Sionna resolves, i.e. bsdf id minus "mat-").
ROLE_MATERIAL = {
    "wall":   "itu_concrete",
    "roof":   "itu_brick",
    "ground": "itu_medium_dry_ground",
}

# ITU-R P.2040-1 coefficients: eps_r = a*f_GHz**b, sigma = c*f_GHz**d.
# Feed straight to cfg["MATERIAL_PARAMS"]. Every entry MUST set
# scattering_coefficient -- build_scene_batch defaults it to 0, which silently
# disables diffuse scattering for that surface.
#
# Measured bands: concrete 1-100 GHz, brick 1-40, medium_dry_ground 1-10 ONLY.
# Ground is therefore extrapolated at 20 GHz (via itu_range_patch.widen(),
# called from build_scene_batch) and is the one extrapolated surface in this
# scene -- state that wherever results are reported.
MATERIAL_PARAMS = {
    "itu_concrete":          {"a": 5.24, "b": 0.0,  "c": 0.0462, "d": 0.7822,
                              "scattering_coefficient": 0.40, "xpd_coefficient": 0.0},
    "itu_brick":             {"a": 3.91, "b": 0.0,  "c": 0.0238, "d": 0.16,
                              "scattering_coefficient": 0.50, "xpd_coefficient": 0.0},
    "itu_medium_dry_ground": {"a": 15.0, "b": -0.1, "c": 0.035,  "d": 1.63,
                              "scattering_coefficient": 0.30, "xpd_coefficient": 0.0},
}

# Visual reflectance only; Sionna discards it when it swaps in the radio material.
_MAT_RGB = {
    "itu_concrete":          (0.539, 0.539, 0.539),
    "itu_brick":             (0.472, 0.258, 0.191),
    "itu_medium_dry_ground": (0.372, 0.334, 0.253),
}

# A face counts as horizontal-up (roof-like) at most 30 deg off horizontal.
ROOF_NORMAL_COS = float(np.cos(np.radians(30.0)))
# ... and as vertical (wall-like) within 5.7 deg of vertical.
WALL_NORMAL_MAX = 0.1

# Vertex weld tolerance [m] for the hole analysis. Each surface is a separate
# PLY with its own vertex list, so shared edges only appear after welding.
WELD_TOL = 1e-3
# Boundary edges with both ends below this are the expected open bottoms.
GROUND_Z_TOL = 0.05
# Mesh + shape id for the generated hole caps.
CAP_MESH = "building_caps.ply"
CAP_SHAPE_ID = "elm__caps"

# PLY scalar type -> (struct char, numpy dtype).
_PLY_SCALAR = {
    "float":  ("f", "<f4"), "float32": ("f", "<f4"),
    "double": ("d", "<f8"), "float64": ("d", "<f8"),
    "int":    ("i", "<i4"), "int32":   ("i", "<i4"),
    "uint":   ("I", "<u4"), "uint32":  ("I", "<u4"),
    "short":  ("h", "<i2"), "int16":   ("h", "<i2"),
    "ushort": ("H", "<u2"), "uint16":  ("H", "<u2"),
    "char":   ("b", "<i1"), "int8":    ("b", "<i1"),
    "uchar":  ("B", "<u1"), "uint8":   ("B", "<u1"),
}


def read_ply(path: str):
    """Read a triangle PLY -> (verts [N,3] float64, faces [M,3] int).

    Handles ascii and binary_little_endian with arbitrary extra vertex
    properties (nx/ny/nz, s/t, Col_*), which all three layouts in this scene
    use. Non-triangular faces raise rather than being silently fanned.
    """
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")

        fmt = None
        elements = []            # [(name, count, [prop, ...])]
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"{path}: header ended without end_header")
            tok = raw.decode("ascii", "replace").strip().split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                elements.append((tok[1], int(tok[2]), []))
            elif tok[0] == "property":
                if not elements:
                    raise ValueError(f"{path}: property before element")
                if tok[1] == "list":
                    elements[-1][2].append(("list", tok[2], tok[3], tok[4]))
                else:
                    elements[-1][2].append(("scalar", tok[1], tok[2]))
            elif tok[0] == "end_header":
                break

        if fmt not in ("ascii", "binary_little_endian"):
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")
        binary = fmt == "binary_little_endian"
        body = fh.read() if binary else fh.read().decode("ascii", "replace").split()

    verts = faces = None
    off = 0                      # byte offset (binary) or token index (ascii)

    for name, count, props in elements:
        if binary and all(p[0] == "scalar" for p in props):
            # Fixed-stride element: read it as one structured array.
            dt = np.dtype([(p[2], _PLY_SCALAR[p[1]][1]) for p in props])
            arr = np.frombuffer(body, dtype=dt, count=count, offset=off)
            off += count * dt.itemsize
            if name == "vertex":
                missing = [c for c in ("x", "y", "z") if c not in dt.names]
                if missing:
                    raise ValueError(f"{path}: vertex element lacks {missing}")
                verts = np.column_stack(
                    [arr["x"], arr["y"], arr["z"]]).astype(np.float64)
            continue

        rows = []
        for _ in range(count):
            row = []
            for p in props:
                if p[0] == "list":
                    if binary:
                        cch = _PLY_SCALAR[p[1]][0]
                        n = struct.unpack_from("<" + cch, body, off)[0]
                        off += struct.calcsize(cch)
                        ich = _PLY_SCALAR[p[2]][0]
                        row = list(struct.unpack_from("<" + ich * n, body, off))
                        off += struct.calcsize(ich) * n
                    else:
                        n = int(body[off]); off += 1
                        row = [int(body[off + k]) for k in range(n)]
                        off += n
                elif binary:
                    off += struct.calcsize(_PLY_SCALAR[p[1]][0])
                else:
                    row.append(float(body[off])); off += 1
            rows.append(row)

        if name == "vertex" and not binary:
            verts = np.asarray([r[:3] for r in rows], dtype=np.float64)
        elif name == "face":
            if any(len(r) != 3 for r in rows):
                raise ValueError(f"{path}: non-triangular face present")
            faces = np.asarray(rows, dtype=np.int64)

    if verts is None or faces is None:
        raise ValueError(f"{path}: missing vertex or face element")
    return verts, faces


def face_normals_z(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit face-normal z component; NaN for degenerate (zero-area) faces."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, n[:, 2] / norm, np.nan)


def _orientation_census(nz: np.ndarray) -> dict:
    """Face-orientation breakdown used to audit how pure a role assignment is."""
    a = np.abs(np.nan_to_num(nz, nan=0.0))
    return {
        "n_faces":    int(len(nz)),
        "n_up":       int((np.nan_to_num(nz, nan=0.0) >= ROOF_NORMAL_COS).sum()),
        "n_down":     int((np.nan_to_num(nz, nan=0.0) <= -ROOF_NORMAL_COS).sum()),
        "n_vertical": int((a < WALL_NORMAL_MAX).sum()),
        "n_sloped":   int(((a >= WALL_NORMAL_MAX) & (a < ROOF_NORMAL_COS)).sum()),
        "n_degenerate": int(np.isnan(nz).sum()),
    }


def _bsdf(root, mat_name):
    """Append a top-level diffuse bsdf, matching the source scene's own style."""
    r, g, b = _MAT_RGB.get(mat_name, (0.5, 0.5, 0.5))
    bid = f"mat-{mat_name}"
    bs = ET.SubElement(root, "bsdf", {"type": "diffuse", "id": bid, "name": bid})
    ET.SubElement(bs, "rgb", {"value": f"{r:.6f} {g:.6f} {b:.6f}",
                              "name": "reflectance"})
    return bs


def _shape(root, shape_id, mesh_rel, mat_name):
    sh = ET.SubElement(root, "shape", {"type": "ply", "id": shape_id, "name": shape_id})
    ET.SubElement(sh, "string", {"name": "filename", "value": mesh_rel})
    ET.SubElement(sh, "boolean", {"name": "face_normals", "value": "true"})
    ET.SubElement(sh, "ref", {"id": f"mat-{mat_name}", "name": "bsdf"})
    return sh


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


def weld_scene(mesh_paths, tol: float = WELD_TOL):
    """Concatenate meshes and weld coincident vertices -> (verts, faces).

    The scene stores each surface as a separate PLY with its own vertex list, so
    a shared edge between two meshes is only recognisable after welding by
    position. Degenerate triangles produced by the weld are dropped.
    """
    V, F, off = [], [], 0
    for p in mesh_paths:
        v, f = read_ply(p)
        V.append(v); F.append(f + off); off += len(v)
    V, F = np.vstack(V), np.vstack(F)

    _, inv = np.unique(np.round(V / tol).astype(np.int64), axis=0, return_inverse=True)
    Vw = np.zeros((int(inv.max()) + 1, 3))
    Vw[inv] = V
    Fw = inv[F]
    keep = ((Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2]))
    return Vw, Fw[keep]


def find_hole_loops(verts, faces, ground_z: float = GROUND_Z_TOL):
    """Closed boundary loops above ground = actual holes in the scene shell."""
    cnt = {}
    for t in faces:
        for a, c in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            e = (a, c) if a < c else (c, a)
            cnt[e] = cnt.get(e, 0) + 1

    bnd = [e for e, n in cnt.items() if n == 1
           and not (verts[e[0], 2] < ground_z and verts[e[1], 2] < ground_z)]

    adj = {}
    for a, c in bnd:
        adj.setdefault(a, []).append(c)
        adj.setdefault(c, []).append(a)

    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, cur, prev, closed = [start], start, None, False
        seen.add(start)
        while True:
            nxt = next((n for n in adj[cur] if n != prev and n not in seen), None)
            if nxt is None:
                closed = start in adj[cur] and len(loop) > 2
                break
            loop.append(nxt); seen.add(nxt); prev, cur = cur, nxt
        loops.append({"idx": loop, "closed": closed,
                      "z": float(verts[loop, 2].mean()),
                      "planar": bool(np.ptp(verts[loop, 2]) < 1e-6)})
    return loops, len(bnd)


def _signed_area(p3):
    x, y = p3[:, 0], p3[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ear_clip(p3):
    """Triangulate a simple (possibly non-convex) polygon by ear clipping.

    Takes the loop's own 3D vertices and works in XY. Every polygon edge ends
    up in exactly one triangle, which is the property that matters here: a cap
    must consume the boundary edges it was built from, or it just trades one
    hole for another.
    """
    p = p3[:, :2]
    n = len(p)
    if n < 3:
        return np.zeros((0, 3), dtype=int)
    idx = list(range(n))
    if _signed_area(p3) < 0:
        idx.reverse()

    def cross(a, b, c):
        return ((p[b][0] - p[a][0]) * (p[c][1] - p[a][1])
                - (p[b][1] - p[a][1]) * (p[c][0] - p[a][0]))

    def inside(a, b, c, q):
        d1, d2, d3 = cross(a, b, q), cross(b, c, q), cross(c, a, q)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    tris, guard = [], 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if cross(a, b, c) <= 0:
                continue                      # reflex vertex, not an ear
            if any(inside(a, b, c, q) for q in idx if q not in (a, b, c)):
                continue                      # another vertex sits in the ear
            tris.append((a, b, c))
            idx.pop(k)
            break
        else:
            break                             # no ear found; emit what is left
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.asarray(tris, dtype=int)


def _stitch_rings(A, B):
    """Triangle strip spanning the gap between two closed coplanar loops."""
    if _signed_area(A) * _signed_area(B) < 0:
        B = B[::-1]
    nA, nB = len(A), len(B)
    d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    i0, j0 = np.unravel_index(int(np.argmin(d)), d.shape)

    tris, i, j = [], 0, 0
    while i < nA or j < nB:
        ai, bj = (i0 + i) % nA, (j0 + j) % nB
        ai1, bj1 = (i0 + i + 1) % nA, (j0 + j + 1) % nB
        take_a = i < nA and (j >= nB or
                             np.linalg.norm(A[ai1] - B[bj])
                             <= np.linalg.norm(A[ai] - B[bj1]))
        if take_a:
            tris.append((ai, nA + bj, ai1)); i += 1
        else:
            tris.append((ai, nA + bj, nA + bj1)); j += 1
    return np.vstack([A, B]), np.asarray(tris, dtype=int)


def _triangulate_level(rings):
    """Close one coplanar set of hole loops."""
    if len(rings) == 1:
        return rings[0], _ear_clip(rings[0])
    if len(rings) == 2:
        return _stitch_rings(rings[0], rings[1])
    raise NotImplementedError(
        f"{len(rings)} coplanar hole loops at one level; only 1 (fill) and "
        f"2 (stitch) are handled")


def _covered_xy(pts_xy, z, V, F, z_tol=0.05):
    """Mask of sample points already covered by scene geometry at height ~z."""
    zc = V[F][:, :, 2]
    cand = F[(np.abs(zc.mean(axis=1) - z) < z_tol)
             & (np.ptp(zc, axis=1) < z_tol)]
    if not len(cand):
        return np.zeros(len(pts_xy), dtype=bool)

    T = V[cand][:, :, :2]                       # (M,3,2)
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    p = pts_xy[:, None, :]                      # (N,1,2)

    def side(u, v):
        return ((v[:, 0] - u[:, 0]) * (p[..., 1] - u[:, 1])
                - (v[:, 1] - u[:, 1]) * (p[..., 0] - u[:, 0]))

    d1, d2, d3 = side(a, b), side(b, c), side(c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return (~(neg & pos)).any(axis=1)


def _redundant_mask(cap_V, cap_F, z, V, F):
    """Which of one level's cap triangles cover already-surfaced area.

    Samples the centroid plus three inset edge midpoints; a triangle counts as
    redundant only when every sample is already covered.
    """
    P = cap_V[cap_F]
    cen = P.mean(axis=1)
    samples = [cen[:, :2]]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        samples.append((0.75 * 0.5 * (P[:, i] + P[:, j]) + 0.25 * cen)[:, :2])

    redundant = np.ones(len(cap_F), dtype=bool)
    for s in samples:
        redundant &= _covered_xy(s, z, V, F)
    return redundant


def build_caps(verts, faces, loops, z_tol: float = 0.05):
    """Cap closed planar hole loops with horizontal polygons.

    Loops are grouped by height and each level is filled under the even-odd
    rule (see _triangulate_level). Returns (cap_verts, cap_faces, report, skipped).
    """
    usable = [l for l in loops if l["closed"] and l["planar"]]
    skipped = [l for l in loops if not (l["closed"] and l["planar"])]

    levels = {}
    for l in usable:
        levels.setdefault(round(l["z"] / z_tol) * z_tol, []).append(l)

    V_out, F_out, report = [], [], []
    for _, group in sorted(levels.items()):
        # Carry the loops' OWN 3D vertices through: synthesising a z from the
        # quantised level key puts the cap ~2 cm off the rim it is closing,
        # which is outside WELD_TOL and leaves the hole open.
        rings = [verts[l["idx"]] for l in group]
        pts, tris = _triangulate_level(rings)
        if not len(tris):
            continue
        z_true = float(np.mean([r[:, 2].mean() for r in rings]))
        red = _redundant_mask(pts, tris, z_true, verts, faces)
        kept = tris[~red]
        report.append({"z_m": round(z_true, 3), "rings": len(rings),
                       "ring_vertices": [int(len(r)) for r in rings],
                       "triangles": int(len(kept)),
                       "dropped_redundant": int(red.sum()),
                       "verdict": "hole closed" if len(kept) else
                                  "seam only (already surfaced)"})
        if not len(kept):
            continue
        base = sum(len(v) for v in V_out)
        V_out.append(pts)
        F_out.append(kept + base)

    if not V_out:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), report, skipped

    V, F = np.vstack(V_out), np.vstack(F_out)
    # Wind every cap outward (up), matching the source scene's convention --
    # its roof faces are 1294 up / 0 down. Stitching a ring pair otherwise
    # emits whichever winding the traversal happened to produce.
    flip = np.nan_to_num(face_normals_z(V, F), nan=1.0) < 0
    F[flip] = F[flip][:, ::-1]
    return V, F, report, skipped


def build_du_scene(src_xml: str = DEFAULT_SRC,
                   out_dir: str = DEFAULT_OUT,
                   scene_name: str = "canarywharf3_du",
                   cap_holes: bool = True) -> dict:
    """Write the three-role Dense Urban scene. Returns (and saves) a report dict."""
    src_dir = os.path.dirname(os.path.abspath(src_xml))
    mesh_out = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_out, exist_ok=True)

    src_root = ET.parse(src_xml).getroot()

    out_root = ET.Element("scene", {"version": src_root.get("version", "2.1.0")})
    for integ in src_root.findall("integrator"):
        out_root.append(integ)
    for mat in ROLE_MATERIAL.values():
        _bsdf(out_root, mat)

    census = []
    n_shape = {r: 0 for r in ROLE_MATERIAL}
    n_face = {r: 0 for r in ROLE_MATERIAL}
    nz_by_role = {r: [] for r in ROLE_MATERIAL}

    for sh in src_root.findall("shape"):
        shape_id = sh.get("id") or ""
        fn_el = sh.find("string[@name='filename']")
        ref_el = sh.find("ref")
        if fn_el is None or ref_el is None:
            raise ValueError(f"{shape_id}: shape has no filename or material ref")
        mesh_rel = fn_el.get("value")
        mesh_base = os.path.basename(mesh_rel)
        src_mat = ref_el.get("id")

        # Ground first: it references the building material in the source, so
        # only the mesh name separates it from the walls.
        if mesh_base == GROUND_MESH:
            role = "ground"
        elif src_mat in SRC_MATERIAL_ROLE:
            role = SRC_MATERIAL_ROLE[src_mat]
        else:
            raise ValueError(f"{shape_id}: unexpected source material {src_mat!r}")

        src_mesh = os.path.join(src_dir, mesh_rel)
        shutil.copyfile(src_mesh, os.path.join(mesh_out, mesh_base))
        _shape(out_root, shape_id, f"meshes/{mesh_base}", ROLE_MATERIAL[role])

        v, f = read_ply(src_mesh)
        nz_by_role[role].append(face_normals_z(v, f))
        n_shape[role] += 1
        n_face[role] += len(f)
        census.append({"shape": shape_id, "mesh": mesh_base, "role": role,
                       "source_material": src_mat,
                       "material": ROLE_MATERIAL[role], "faces": int(len(f))})

    # ---- close the shell -----------------------------------------------------
    # Buildings are open shells: rays from a satellite near zenith otherwise
    # enter through the gaps. Caps are horizontal polygons over the closed
    # planar boundary loops, and take the roof material.
    src_meshes = [os.path.join(src_dir, os.path.basename(fn.get("value")))
                  for fn in src_root.findall("shape/string[@name='filename']")]
    src_meshes = [os.path.join(src_dir, "meshes", os.path.basename(p))
                  for p in src_meshes]
    Vw, Fw = weld_scene(src_meshes)
    loops, n_bnd = find_hole_loops(Vw, Fw)

    caps = {"enabled": cap_holes, "n_boundary_edges": n_bnd,
            "n_loops": len(loops), "levels": [], "skipped_loops": [],
            "n_triangles": 0, "residual_boundary_edges": None,
            "residual_geometric_holes": None}

    if cap_holes and loops:
        cv, cf, rep, skipped = build_caps(Vw, Fw, loops)
        caps["levels"] = rep
        caps["skipped_loops"] = [{"z": round(l["z"], 2), "n": len(l["idx"]),
                                  "closed": l["closed"], "planar": l["planar"]}
                                 for l in skipped]
        if len(cf):
            write_ply(os.path.join(mesh_out, CAP_MESH), cv, cf)
            _shape(out_root, CAP_SHAPE_ID, f"meshes/{CAP_MESH}", ROLE_MATERIAL["roof"])
            n_shape["roof"] += 1
            n_face["roof"] += len(cf)
            caps["n_triangles"] = int(len(cf))
            nz_by_role["roof"].append(face_normals_z(cv, cf))
            census.append({"shape": CAP_SHAPE_ID, "mesh": CAP_MESH, "role": "roof",
                           "source_material": "(generated cap)",
                           "material": ROLE_MATERIAL["roof"], "faces": int(len(cf))})
            # Re-run the analysis with the caps in place. Leftover boundary
            # edges are expected -- the seam levels are deliberately not capped
            # -- so the assertion that matters is that no loop still bounds
            # uncovered area, i.e. re-capping would produce nothing.
            Vc, Fc = weld_scene(src_meshes + [os.path.join(mesh_out, CAP_MESH)])
            loops2, caps["residual_boundary_edges"] = find_hole_loops(Vc, Fc)
            _, cf2, _, _ = build_caps(Vc, Fc, loops2)
            caps["residual_geometric_holes"] = int(len(cf2))

    _indent(out_root)
    out_xml = os.path.join(out_dir, f"{scene_name}.xml")
    ET.ElementTree(out_root).write(out_xml, encoding="utf-8", xml_declaration=False)

    purity = {r: _orientation_census(np.concatenate(v)) if v else {}
              for r, v in nz_by_role.items()}

    report = {
        "source_xml": os.path.abspath(src_xml),
        "scene_xml": os.path.abspath(out_xml),
        "mesh_dir": os.path.abspath(mesh_out),
        "source_material_role": SRC_MATERIAL_ROLE,
        "role_material": ROLE_MATERIAL,
        "material_params": MATERIAL_PARAMS,
        "material_names": sorted(ROLE_MATERIAL.values()),
        "n_shapes": n_shape,
        "n_faces": n_face,
        "n_shapes_total": sum(n_shape.values()),
        "n_faces_total": sum(n_face.values()),
        "dropped_materials": ["mat-itu_metal"],
        "role_purity": purity,
        "caps": caps,
        "shape_census": census,
    }
    meta_path = os.path.join(out_dir, f"{scene_name}_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(report, fh, indent=2)
    report["meta_json"] = os.path.abspath(meta_path)

    print(f"[build_wharf_du_scene] wrote {out_xml}")
    print(f"  shapes : {report['n_shapes_total']}  {n_shape}")
    print(f"  faces  : {report['n_faces_total']}  {n_face}")
    for r, p in purity.items():
        if p:
            print(f"  {r:6s} -> {ROLE_MATERIAL[r]:22s} "
                  f"vertical={p['n_vertical']:5d} up={p['n_up']:5d} "
                  f"sloped={p['n_sloped']:4d} down={p['n_down']:4d}")
    print(f"  materials : {report['material_names']}  (dropped mat-itu_metal)")
    if caps["enabled"]:
        lv = "; ".join(f"z={l['z_m']}m: {l['triangles']} tri kept, "
                       f"{l['dropped_redundant']} redundant -> {l['verdict']}"
                       for l in caps["levels"])
        print(f"  caps   : {caps['n_triangles']} triangles over "
              f"{len(caps['levels'])} level(s)  [{lv}]")
        seams = [l for l in caps["levels"] if not l["triangles"]]
        holes = caps["residual_geometric_holes"]
        if holes == 0:
            print(f"  shell closed: no uncovered openings remain "
                  f"({caps['residual_boundary_edges']} boundary edges left, all "
                  f"seams at {len(seams)} already-surfaced level(s))")
        elif holes:
            print(f"  WARNING: {holes} cap triangles' worth of opening remains "
                  f"after capping")
        if caps["skipped_loops"]:
            print(f"  WARNING: {len(caps['skipped_loops'])} loops not capped "
                  f"(open or non-planar); see caps.skipped_loops in the meta JSON")
    elif caps["n_boundary_edges"]:
        print(f"  WARNING: capping disabled; {caps['n_boundary_edges']} boundary "
              f"edges above ground (buildings are open shells)")
    print(f"[build_wharf_du_scene] wrote {meta_path}")
    return report


def _cli():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", default=DEFAULT_SRC, help="source canarywharf3.xml")
    p.add_argument("--out_dir", default=DEFAULT_OUT, help="output scene directory")
    p.add_argument("--scene_name", default="canarywharf3_du")
    p.add_argument("--no_caps", action="store_true",
                   help="leave the buildings as open shells")
    a = p.parse_args()
    build_du_scene(a.src, a.out_dir, a.scene_name, cap_holes=not a.no_caps)


if __name__ == "__main__":
    _cli()
