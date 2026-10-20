"""Dependency-free PLY reader and binary STL writer (no trimesh in the venv)."""

import numpy as np

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path):
    """Return (vertices Nx3 float64, faces Mx3 int64) with faces fan-triangulated."""
    with open(path, "rb") as fh:
        raw = fh.read()

    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: no end_header")
    head = raw[:end].decode("ascii", "replace").splitlines()
    body = raw[raw.find(b"\n", end) + 1:]

    fmt = None
    elements = []  # (name, count, [(prop, ...)])
    for line in head:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property" and elements:
            elements[-1][2].append(tok[1:])

    if fmt not in ("binary_little_endian", "ascii"):
        raise ValueError(f"{path}: unsupported PLY format {fmt}")
    if fmt == "ascii":
        return _read_ascii(body, elements)

    verts, faces, off = None, None, 0
    for name, count, props in elements:
        if props and props[0][0] == "list":
            faces, off = _read_binary_faces(body, off, count, props[0])
            if name != "face":
                faces = None
            continue
        dt = np.dtype([(p[1], "<" + _PLY_TYPES[p[0]]) for p in props])
        block = np.frombuffer(body, dtype=dt, count=count, offset=off)
        off += count * dt.itemsize
        if name == "vertex":
            verts = np.stack([block["x"], block["y"], block["z"]], 1).astype(np.float64)

    if verts is None or faces is None:
        raise ValueError(f"{path}: missing vertex or face element")
    return verts, faces


def _read_binary_faces(body, off, count, prop):
    """Parse a binary list property; fast path when every face has the same arity."""
    cdt = np.dtype("<" + _PLY_TYPES[prop[1]])
    idt = np.dtype("<" + _PLY_TYPES[prop[2]])
    k = int(np.frombuffer(body, dtype=cdt, count=1, offset=off)[0])
    stride = cdt.itemsize + k * idt.itemsize
    if len(body) - off >= count * stride:
        rec = np.dtype([("n", cdt), ("v", idt, (k,))])
        block = np.frombuffer(body, dtype=rec, count=count, offset=off)
        if np.all(block["n"] == k):
            return _fan(block["v"].astype(np.int64)), off + count * stride

    tris, pos = [], off  # ragged fallback
    for _ in range(count):
        n = int(np.frombuffer(body, dtype=cdt, count=1, offset=pos)[0])
        pos += cdt.itemsize
        idx = np.frombuffer(body, dtype=idt, count=n, offset=pos).astype(np.int64)
        pos += n * idt.itemsize
        tris.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, n - 1))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3), pos


def _read_ascii(body, elements):
    lines = body.decode("ascii", "replace").split("\n")
    verts, tris, ln = [], [], 0
    for name, count, props in elements:
        for _ in range(count):
            while not lines[ln].strip():
                ln += 1
            tok = lines[ln].split()
            ln += 1
            if name == "vertex":
                verts.append([float(t) for t in tok[:3]])
            elif name == "face":
                idx = [int(t) for t in tok[1:1 + int(tok[0])]]
                tris.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, len(idx) - 1))
    return np.asarray(verts, np.float64), np.asarray(tris, np.int64).reshape(-1, 3)


def _fan(quads):
    """Fan-triangulate a uniform-arity face block."""
    k = quads.shape[1]
    if k == 3:
        return quads
    return np.concatenate([quads[:, [0, j, j + 1]] for j in range(1, k - 1)], axis=0)


def write_stl(path, verts, faces):
    """Write binary STL. Normals are recomputed; degenerate facets get a zero normal."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, mag, out=np.zeros_like(n), where=mag > 0)

    rec = np.zeros(len(faces), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as fh:
        fh.write(b"ntnrt merged scene".ljust(80, b"\0"))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())
