"""Fetch and cache the Celestrak bulk TLE catalog under tle_catalog/.
"""

import time
from pathlib import Path

import requests

_CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php"


def fetch_tle_catalog(group: str,
                       cache_dir: str = "./tle_catalog",
                       max_age_hours: float = 6.0,
                       force: bool = False) -> str:
    """Download (or reuse a fresh cached copy of) a bulk TLE catalog for `group`
(a celestrak GROUP name, e.g. "starlink", "oneweb", "active").
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{group}.tle"

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        age_hours = (time.time() - out_path.stat().st_mtime) / 3600.0
        if age_hours <= max_age_hours:
            print(f"[fetch_tle] Cache hit: {out_path.name} (age {age_hours:.1f} h)")
            return str(out_path)
        print(f"[fetch_tle] Cache stale ({age_hours:.1f} h > {max_age_hours} h) — re-fetching")

    url = f"{_CELESTRAK_BASE}?GROUP={group}&FORMAT=TLE"
    print(f"[fetch_tle] Fetching: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.text

    if not text.strip() or "<html" in text.lower()[:200]:
        raise RuntimeError(
            f"celestrak returned no usable TLE data for GROUP='{group}'. "
            f"Check the group name against https://celestrak.org/NORAD/elements/ "
            f"(common values: starlink, oneweb, active, gps-ops)."
        )

    n_lines = len([ln for ln in text.splitlines() if ln.strip()])
    if n_lines % 3 != 0:
        raise RuntimeError(
            f"celestrak response for GROUP='{group}' has {n_lines} non-empty "
            f"lines, not a multiple of 3 — unexpected TLE format."
        )

    out_path.write_text(text)
    print(f"[fetch_tle] Downloaded {n_lines // 3} satellites -> {out_path}")
    return str(out_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch a bulk celestrak TLE catalog and cache it locally"
    )
    parser.add_argument("--group", type=str, required=True,
                         help="celestrak GROUP name, e.g. starlink, oneweb, active")
    parser.add_argument("--cache_dir", type=str, default="./tle_catalog")
    parser.add_argument("--max_age_hours", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = fetch_tle_catalog(args.group, args.cache_dir, args.max_age_hours, args.force)
    print(f"\nTLE_CATALOG_FILE = \"{path}\"")
