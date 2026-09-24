"""
Verify the loaded Sionna carries this project's patches.

Replaces the retired sionna_shadow_patch.py: the epsilon guard now lives in the
patched source (sionna_patch/), so there is nothing to apply at runtime -- only
something to assert. Nothing here imports CUDA or builds a scene; scene_utils
keeps Sionna off its import path deliberately.

See docs/sionna-patches.md.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re

import sionna
import sionna.rt
from sionna.rt.constants import EPSILON_FLOAT

SUPPORTED_VERSION = "1.2.1"
ARCHIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sionna_patch")
ALLOW_OTHER_VERSION = os.environ.get("RUS_ALLOW_SIONNA_VERSION") == "1"

# The guard expression the epsilon patch installs, for corroborating the constant.
_GUARD_RE = re.compile(r"dr\.minimum\(\s*maxt\s*\*\s*EPSILON_FLOAT\s*,\s*MAX_RAY_GUARD_M")


class PatchMissingError(RuntimeError):
    """Raised when a required Sionna patch is absent from the loaded package."""


def dead_zone_m(ray_length_m: float, patched: bool = True) -> float:
    """Depth of the unchecked region at the ray's target end [m]."""
    g = ray_length_m * EPSILON_FLOAT
    if not patched:
        return g
    cap = getattr(_ray_tracing(), "MAX_RAY_GUARD_M", 1e-3)
    return min(g, cap)


def _ray_tracing():
    import sionna.rt.utils.ray_tracing as rt
    return rt


def _check_beamforming() -> dict:
    """Signature introspection: cheap, no side effects, no GPU."""
    try:
        params = inspect.signature(sionna.rt.PathSolver.__call__).parameters
    except (AttributeError, TypeError, ValueError) as exc:
        return {"ok": False, "reason": f"could not introspect PathSolver.__call__: {exc}"}
    missing = [p for p in ("beamforming", "beam_angle") if p not in params]
    if missing:
        return {"ok": False, "reason": f"PathSolver.__call__ missing {missing}"}
    return {"ok": True, "reason": "PathSolver.__call__ accepts beamforming, beam_angle"}


def _check_epsilon_guard() -> dict:
    """Constant is the gate; the source expression corroborates it.

    The constant alone could exist unused, and source alone breaks on a
    .pyc-only install, so a pass needs the constant plus either a source match
    or an explicit note that the source was unreadable.
    """
    rt = _ray_tracing()
    cap = getattr(rt, "MAX_RAY_GUARD_M", None)
    if cap is None:
        return {"ok": False, "reason": "ray_tracing.MAX_RAY_GUARD_M is absent"}
    try:
        src = inspect.getsource(rt.spawn_ray_to)
    except (OSError, TypeError):
        return {"ok": True, "cap_m": cap, "reason": "MAX_RAY_GUARD_M present",
                "note": "source_unavailable: expression not corroborated"}
    if not _GUARD_RE.search(src):
        return {"ok": False, "cap_m": cap,
                "reason": "MAX_RAY_GUARD_M present but spawn_ray_to does not use it"}
    return {"ok": True, "cap_m": cap, "reason": "guard capped in spawn_ray_to"}


def _check_manifest() -> dict:
    """Compare installed files against sionna_patch/MANIFEST.json.

    Catches the half-applied copy, which introspection cannot: one file copied
    into site-packages and another not.
    """
    path = os.path.join(ARCHIVE, "MANIFEST.json")
    if not os.path.exists(path):
        return {"ok": None, "reason": f"no manifest at {path}"}
    man = json.load(open(path))
    root = os.path.dirname(os.path.dirname(os.path.abspath(sionna.__file__)))
    stale = []
    for entry in man["files"]:
        live = os.path.join(root, entry["path"])
        if not os.path.exists(live):
            stale.append(f"{entry['path']} (absent)")
            continue
        digest = hashlib.sha256(open(live, "rb").read()).hexdigest()
        if digest != entry["sha256_patched"]:
            which = "pristine" if digest == entry["sha256_pristine"] else "unknown"
            stale.append(f"{entry['path']} ({which})")
    if stale:
        return {"ok": False, "reason": "installed files differ from archive: "
                                       + ", ".join(stale)}
    return {"ok": True, "reason": f"all {len(man['files'])} files match the archive"}


def _cuda_available() -> bool:
    # ray_tracing.py hardcodes dr.cuda.ad.Float in the cone path, so on an
    # LLVM-only build the beamforming patch is present but cannot run.
    try:
        import drjit as dr
        return bool(dr.has_backend(dr.JitBackend.CUDA))
    except Exception:
        return False


def check() -> dict:
    """Full patch report. Never raises."""
    version = getattr(sionna, "__version__", "unknown")
    supported = version == SUPPORTED_VERSION
    beam = _check_beamforming()
    eps = _check_epsilon_guard()
    manifest = _check_manifest()
    rt_patch = getattr(_ray_tracing(), "__RUS_PATCH__", None)
    return {
        "ok": bool(beam["ok"] and eps["ok"] and manifest["ok"] is not False
                   and (supported or ALLOW_OTHER_VERSION)),
        "sionna_version": version,
        "version_supported": supported,
        "beamforming": beam,
        "epsilon_guard": eps,
        "manifest": manifest,
        "archive_rev": (rt_patch or {}).get("rev"),
        "cuda_available": _cuda_available(),
    }


def summary_line(report: dict) -> str:
    """One line for a run log."""
    eps = report["epsilon_guard"]
    cap = eps.get("cap_m")
    cap_s = f"{cap * 1e3:.1f} mm" if cap else "ABSENT"
    return (f"sionna {report['sionna_version']}: beamforming="
            f"{'yes' if report['beamforming']['ok'] else 'NO'} "
            f"guard={cap_s} cuda={'yes' if report['cuda_available'] else 'no'} "
            f"archive={report['archive_rev'] or '?'}")


def require(context: str = "") -> dict:
    """Assert both patches are present; raise PatchMissingError otherwise."""
    report = check()
    where = f" (required by {context})" if context else ""

    if not report["version_supported"]:
        msg = (f"sionna {report['sionna_version']} found, expected "
               f"{SUPPORTED_VERSION}. The archived patches are line-matched to "
               f"{SUPPORTED_VERSION}; re-derive them from sionna_patch/diffs/ "
               f"before trusting results{where}.")
        if not ALLOW_OTHER_VERSION:
            raise PatchMissingError(msg)
        print(f"[sionna_patch_check] WARNING: {msg} "
              f"(RUS_ALLOW_SIONNA_VERSION=1 set)")

    broken = [name for name in ("beamforming", "epsilon_guard")
              if not report[name]["ok"]]
    if broken:
        raise PatchMissingError(
            f"Sionna patches missing: {', '.join(broken)}{where}. "
            + "; ".join(f"{n}: {report[n]['reason']}" for n in broken)
            + ". Re-apply from sionna_patch/ -- see its README. Without the "
              "epsilon guard the occlusion dead zone corrupts the LoS state.")

    if report["manifest"]["ok"] is False:
        raise PatchMissingError(
            f"Sionna install does not match sionna_patch/MANIFEST.json{where}: "
            f"{report['manifest']['reason']}. A partially applied patch set is "
            f"worse than none -- re-copy the whole archive.")

    if not report["cuda_available"]:
        print("[sionna_patch_check] WARNING: no CUDA backend; the beamforming "
              "cone path hardcodes dr.cuda.ad.Float and will fail.")
    return report


def _selftest() -> int:
    """
    Scene-level check of the guard against the mesh itself.

    Ground truth is the full-length ray (no guard) tested against the same
    Mitsuba scene the solver uses. The analytic wall test in validate_chiu_roy
    is deliberately not the reference: it treats walls as zero-thickness centre
    lines, so it disagrees on grazing cases that are real hits unrelated to
    this patch.
    """
    import sys
    from pathlib import Path
    import numpy as np
    import mitsuba as mi
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Ray_Tracing.src.utils.orbital_utils import slant_range_m

    mi.set_variant("llvm_ad_rgb")
    xml = "chiu_validation/example_scene_urban/wall_scene.xml"
    if not os.path.exists(xml):
        print(f"[selftest] {xml} not found; run from the repo root")
        return 2
    scene = mi.load_file(xml)
    GT = np.array([0., 0., 1.5])

    def occluded(u, slant, guard_m):
        """Sionna's orientation: origin at the TX, target the RX."""
        tx = GT + u * slant
        d = GT - tx
        L = np.linalg.norm(d)
        d = d / L
        r = mi.Ray3f(mi.Point3f(*[float(x) for x in tx]),
                     mi.Vector3f(*[float(x) for x in d]),
                     maxt=float(L - guard_m), time=0., wavelengths=mi.Color0f())
        return bool(np.array(scene.ray_test(r)).ravel()[0])

    ok = True
    for beta in (10., 40., 80.):
        slant = float(slant_range_m(550., beta))
        old_dz, new_dz = dead_zone_m(slant, False), dead_zone_m(slant, True)
        b = np.radians(beta)
        n = wrong_before = wrong_after = 0
        for az in np.arange(0., 360., 1.):
            a = np.radians(az)
            u = np.array([np.cos(b)*np.cos(a), np.cos(b)*np.sin(a), np.sin(b)])
            truth = occluded(u, slant, 0.0)          # full-length ray = the mesh
            if occluded(u, slant, old_dz) != truth:
                wrong_before += 1
            if occluded(u, slant, new_dz) != truth:
                wrong_after += 1
            n += 1
        good = wrong_after == 0
        ok &= good
        print(f"[selftest] beta={beta:4.0f} deg  slant={slant/1e3:6.0f} km  "
              f"dead zone {old_dz:6.2f} m -> {new_dz*1e3:.1f} mm | "
              f"{n} directions: wrong before {wrong_before:3d}, "
              f"after {wrong_after:3d}  {'OK' if good else 'FAIL'}")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    rep = check()
    print(json.dumps(rep, indent=2))
    print(summary_line(rep))
    raise SystemExit(0 if rep["ok"] else 1)
