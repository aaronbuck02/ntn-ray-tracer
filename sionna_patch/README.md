# Sionna RT patches

Two patches to `sionna-rt 1.2.1` that this project depends on. **These are
reference copies — nothing here installs itself.** If the venv is rebuilt or
`sionna-rt` is reinstalled, both patches are silently lost and the simulations
produce wrong results without erroring. `sionna_patch_check.py` exists to catch
exactly that; every runner calls it before solving.

## The patches

### `beamforming`

Confines the ray budget to a cone about boresight. An isotropic launch from
550 km puts essentially no rays on a 2 km scene, and Mitsuba caps the ray count
at `2**32 - 1`, so the density cannot be recovered by launching more.

Four files cooperate:

| File | Change |
|---|---|
| `rt/utils/ray_tracing.py` | `spawn_ray_from_sources` gains `src_orientations` / `beam_angle`; samples `mi.warp.square_to_uniform_cone` and rotates boresight from +z to +x |
| `rt/path_solvers/path_solver.py` | `beamforming` / `beam_angle` kwargs on `__call__`; routes to the cone generator, forwards `beam_angle` |
| `rt/path_solvers/sb_candidate_generator.py` | threads both through the shoot-and-bounce chain |
| `rt/path_solvers/field_calculator.py` | initial ray-tube solid angle becomes `2*pi*(1-cos(beam_angle))/samples_per_src` instead of `4*pi/samples_per_src`, preserving total transmit power |

**Known limitation:** `ray_tracing.py` hardcodes `dr.cuda.ad.Float` in the cone
call, so this path requires a CUDA Dr.Jit variant. On a CPU/LLVM build the patch
is present but broken — `sionna_patch_check.py` reports CUDA availability for
this reason.

### `epsilon_guard`

Sionna shortens every occlusion ray by a **relative** epsilon,
`EPSILON_FLOAT = 100 * 2**-24 = 5.96e-6` of the ray length
(`rt/utils/ray_tracing.py`, `spawn_ray_to`). At terrestrial range that is
sub-millimetre. Over a 550 km LEO slant it is **10.8 m at 10° elevation**, and
because the direct-path test is spawned from the transmitter toward the
receiver, the shortening lands at the *receiver*: an occluder nearer than the
guard falls outside the ray's search interval, so a **blocked link is reported
as LoS**.

The patch caps the shortening at an absolute `MAX_RAY_GUARD_M = 1e-3`:

```python
maxt = dr.maximum(maxt - dr.minimum(maxt * EPSILON_FLOAT, MAX_RAY_GUARD_M), 0.)
```

Short rays keep their original relative guard; long ones stop losing metres.
`EPSILON_FLOAT` itself is deliberately untouched — it is overloaded elsewhere in
the module. Path lengths, delays and powers are unaffected, being computed from
the vertex geometry rather than the ray extent.

This previously lived as a runtime monkey-patch (`sionna_shadow_patch.py`,
now retired). Folding it into the source removes the import-order hazard that
required rebinding `spawn_ray_to` across four modules.

Derivation and the elevation-vs-dead-zone table: `docs/sionna-patches.md`.

## Layout

```
sionna-1.2.1/     patched files, mirroring the installed package tree
pristine-1.2.1/   the same files straight from the wheel, for diffing
diffs/            unified diff, pristine -> patched, one per file
MANIFEST.json     sha256 of pristine / patched / currently-installed, per file
superseded/       older, mutually incompatible copies found loose on disk
```

Every hunk in `sionna-1.2.1/` carries an inline `# [PATCH: <name>]` marker and
every file a header banner, so a copy of unknown origin can be identified with
`grep -c "\[PATCH" file.py`. The three generations in `superseded/` had no such
markers, which is why they were indistinguishable.

`sionna/__init__.py` is **not** patched. It differs from the `sionna-rt` wheel
only because the `sionna` umbrella package (also 1.2.1) provides its own copy
and wins. That is upstream packaging, not a local edit.

## Re-applying after a reinstall

```bash
SP=$(python -c "import sionna, os; print(os.path.dirname(sionna.__file__))")
cp -r sionna_patch/sionna-1.2.1/rt "$SP"/
python sionna_patch_check.py --check     # must report ok: True
python sionna_patch_check.py --selftest  # exercises the guard end to end
```

The archive is line-matched to 1.2.1. On a different Sionna, do not copy these
files — re-derive the patches from `diffs/` against the new source, since
upstream may have changed or fixed either site.
