"""Diff numeric literals between each MATLAB port and its Python original.

MATLAB cannot be launched in every environment (R2026a needs an interactive
MathWorks sign-in), so this catches the dominant porting risk - a mistyped
coefficient - without executing either side. It is a lint, not a proof: run
tools/parity_check.py once MATLAB is available for the real numeric check.

Usage: python audit_constants.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = Path("/path/to/rician_urban_sim")

# MATLAB port -> Python original it was transcribed from.
PAIRS = [
    ("+ntnrt/+iono/constants.m", "ionospheric_utils.py"),
    ("+ntnrt/+iono/mappingFunction.m", "ionospheric_utils.py"),
    ("+ntnrt/+iono/slantTec.m", "ionospheric_utils.py"),
    ("+ntnrt/+iono/tecLognormalParams.m", "ionospheric_utils.py"),
    ("+ntnrt/+iono/tecPresets.m", "ionospheric_utils.py"),
    ("+ntnrt/+iono/applyIonosphericCorrections.m", "ionospheric_utils.py"),
    ("+ntnrt/+atmos/constants.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/weatherPresets.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/rainPresets.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/wetRefractivity.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/gaseousSpecificAttenuation.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/gaseousAttenuationDb.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/rainCoefficientsP838.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/rainSlantAttenuationDb.m", "atmospheric_utils.py"),
    ("+ntnrt/+atmos/scintillationSigmaDb.m", "atmospheric_utils.py"),
]

NUM = re.compile(r"(?<![\w.])(\d+\.?\d*(?:[eE][-+]?\d+)?)")
# Literals that are structural rather than physical, plus constants the Python
# side imports instead of spelling out (scipy.constants.c).
IGNORE = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 20.0, 100.0, 1e-10, 1e-12, 1e-3,
          299792458.0}


def literals(text, strip_comments):
    out = set()
    for line in text.splitlines():
        if strip_comments == "m" and line.lstrip().startswith("%"):
            continue
        if strip_comments == "py" and line.lstrip().startswith("#"):
            continue
        for tok in NUM.findall(line):
            try:
                v = float(tok)
            except ValueError:
                continue
            if v not in IGNORE:
                out.add(v)
    return out


def close(v, pool, rel=1e-9):
    return any(abs(v - p) <= rel * max(1.0, abs(p)) for p in pool)


def main():
    bad = 0
    for mfile, pyfile in PAIRS:
        mtext = (ROOT / mfile).read_text()
        ptext = (PY / pyfile).read_text()
        mlit = literals(mtext, "m")
        plit = literals(ptext, "py")
        missing = sorted(v for v in mlit if not close(v, plit))
        if missing:
            bad += len(missing)
            print(f"[!] {mfile}")
            for v in missing:
                print(f"      {v:g} not found in {pyfile}")
        else:
            print(f"[ok] {mfile}  ({len(mlit)} literals matched)")

    print(f"\n{bad} unmatched literal(s).")
    print("Unmatched values are not necessarily wrong - check each against the "
          "recommendation text before dismissing it.")


if __name__ == "__main__":
    main()
