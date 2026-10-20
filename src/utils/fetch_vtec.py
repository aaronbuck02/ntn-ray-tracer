"""Download IGS IONEX global ionosphere maps and derive VTEC mean and spread.

Cached under ionex_cache/. Feeds ionospheric_utils, which is off by default.
See docs/physics.md.
"""

import os
import gzip
import shutil
import struct
import datetime
import ftplib
from pathlib import Path
from typing import Optional

import re

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

# ── NASA Earthdata session: re-attaches credentials on cross-host redirects ───
#    Without this, requests strips the Authorization header when CDDIS redirects
#    to urs.earthdata.nasa.gov, causing a 401 / HTML login-page response.
class _EarthdataSession(requests.Session):
    """
    requests.Session subclass that preserves Basic-Auth credentials when
    following redirects to/from urs.earthdata.nasa.gov.

    Based on the NASA Earthdata cookbook:
    https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
    """
    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = HTTPBasicAuth(username, password)

    def rebuild_auth(self, prepared_request, response):
        """Keep Authorization header unless we're leaving both CDDIS and URS."""
        headers = prepared_request.headers
        if "Authorization" in headers:
            orig_host = requests.utils.urlparse(response.request.url).hostname
            redir_host = requests.utils.urlparse(prepared_request.url).hostname
            # Strip only if hopping away from both auth-related hosts
            if orig_host != redir_host \
                    and redir_host != self.AUTH_HOST \
                    and orig_host != self.AUTH_HOST:
                del headers["Authorization"]

# georinex handles RINEX observation/navigation files — NOT IONEX (TEC maps).
# The minimal parser below is used for IONEX regardless of georinex availability.
_HAS_GEORINEX = False

# ── Optional unlzw3 for Unix .Z decompression ─────────────────────────────────
try:
    from unlzw3 import unlzw
    _HAS_UNLZW = True
except ImportError:
    _HAS_UNLZW = False


# ─── IONEX source configuration ───────────────────────────────────────────────

# NASA CDDIS HTTPS endpoint (requires Earthdata .netrc auth)
_CDDIS_BASE = "https://cddis.nasa.gov/archive/gnss/products/ionex"

# BKG anonymous FTP mirror (no account needed, slightly lower reliability)
_BKG_FTP_HOST = "igs.bkg.bund.de"
_BKG_FTP_PATH = "/IGS/products/ionosphere"   # {year}/{doy}/

# IGS combined product prefix
_PRODUCT = "igsg"   # IGS combined solution (most accurate single product)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _doy(date: datetime.date) -> int:
    """Day of year (1-based)."""
    return date.timetuple().tm_yday


def _ionex_filename(date: datetime.date, compressed: bool = True) -> str:
    """
    IGS IONEX filename convention (old-style, pre-2022):
        igsgDDD0.YYi[.Z]
    where DDD = day-of-year, YY = 2-digit year.
    """
    doy = _doy(date)
    yr2 = date.strftime("%y")
    fname = f"{_PRODUCT}{doy:03d}0.{yr2}i"
    return fname + ".Z" if compressed else fname


def _ionex_filename_long(date: datetime.date) -> str:
    """
    IGS IONEX long filename convention (post-2022):
        IGS0OPSFIN_YYYYDDDHHMMDURATIONTECi.ION.gz
    Simplified to the most common daily product.
    """
    doy  = _doy(date)
    year = date.year
    return f"IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX.gz"


def _decompress(src: Path, dst: Path) -> None:
    """Decompress .Z (Unix compress) or .gz file to dst."""
    suffix = src.suffix.lower()
    if suffix == ".gz":
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    elif suffix == ".z":
        if not _HAS_UNLZW:
            raise RuntimeError(
                "unlzw3 is required to decompress .Z files: pip install unlzw3"
            )
        with open(src, "rb") as fin:
            data = unlzw(fin.read())
        with open(dst, "wb") as fout:
            fout.write(data)
    else:
        shutil.copy(src, dst)


# ─── Download functions ────────────────────────────────────────────────────────

def _download_cddis(date: datetime.date, cache_dir: Path) -> Optional[Path]:
    """
    Download IONEX from NASA CDDIS (requires ~/.netrc Earthdata credentials).
    Tries both old-style (.Z) and new long filename (.gz) conventions.
    Returns path to the decompressed .ionex file, or None on failure.
    """
    year = date.year
    doy  = _doy(date)

    candidates = [
        (f"{_CDDIS_BASE}/{year}/{doy:03d}/", _ionex_filename(date, compressed=True)),
        (f"{_CDDIS_BASE}/{year}/{doy:03d}/", _ionex_filename_long(date)),
    ]

    # Auth priority:
    #   1. EARTHDATA_TOKEN env var  → Bearer token (no .netrc needed)
    #   2. ~/.netrc login/password  → Basic auth via redirect-aware session
    session = requests.Session()
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        netrc_path = Path.home() / ".netrc"
        user, pwd = None, None
        if netrc_path.exists():
            import netrc as netrc_lib
            try:
                nrc = netrc_lib.netrc(str(netrc_path))
                creds = nrc.authenticators("urs.earthdata.nasa.gov")
                if creds:
                    user, _, pwd = creds
            except Exception:
                pass
        if user and pwd:
            session = _EarthdataSession(user, pwd)

    for base_url, fname in candidates:
        url      = base_url + fname
        out_comp = cache_dir / fname
        out_ion  = cache_dir / fname.replace(".Z", "").replace(".gz", "")

        if out_ion.exists() and out_ion.stat().st_size > 0:
            print(f"[fetch_vtec] Cache hit: {out_ion.name}")
            return out_ion
        elif out_ion.exists():
            print(f"[fetch_vtec] Stale 0-byte cache file removed: {out_ion.name}")
            out_ion.unlink()

        print(f"[fetch_vtec] Trying CDDIS: {url}")
        try:
            resp = session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            # Detect accidental HTML response (no credentials / redirect failure)
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise RuntimeError(
                    "Got HTML instead of binary data — Earthdata credentials "
                    "missing or invalid. Add urs.earthdata.nasa.gov to ~/.netrc "
                    "(see fetch_vtec.py module docstring)."
                )
            with open(out_comp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            _decompress(out_comp, out_ion)
            out_comp.unlink(missing_ok=True)
            print(f"[fetch_vtec] Downloaded from CDDIS → {out_ion.name}")
            return out_ion
        except Exception as e:
            print(f"[fetch_vtec] CDDIS failed ({e})")
            if out_comp.exists():
                out_comp.unlink()

    return None


def _download_bkg(date: datetime.date, cache_dir: Path) -> Optional[Path]:
    """
    Download IONEX from BKG mirror (anonymous, no credentials needed).
    Tries HTTPS first (works even when port 21 is blocked), then FTP.
    Returns path to decompressed .ionex file, or None on failure.
    """
    year = date.year
    doy  = _doy(date)
    fname     = _ionex_filename(date, compressed=True)
    out_comp  = cache_dir / fname
    out_ion   = cache_dir / fname.replace(".Z", "")

    if out_ion.exists() and out_ion.stat().st_size > 0:
        print(f"[fetch_vtec] Cache hit: {out_ion.name}")
        return out_ion
    elif out_ion.exists():
        print(f"[fetch_vtec] Stale 0-byte cache file removed: {out_ion.name}")
        out_ion.unlink()

    # ── 1. Try BKG HTTPS (no firewall issues, port 443) ──────────────────────
    https_url = (
        f"https://{_BKG_FTP_HOST}{_BKG_FTP_PATH}/{year}/{doy:03d}/{fname}"
    )
    print(f"[fetch_vtec] Trying BKG HTTPS: {https_url}")
    try:
        resp = requests.get(https_url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(out_comp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        _decompress(out_comp, out_ion)
        out_comp.unlink(missing_ok=True)
        print(f"[fetch_vtec] Downloaded from BKG HTTPS → {out_ion.name}")
        return out_ion
    except Exception as e:
        print(f"[fetch_vtec] BKG HTTPS failed ({e})")
        if out_comp.exists():
            out_comp.unlink()

    # ── 2. Fall back to BKG FTP (anonymous) ──────────────────────────────────
    remote_path = f"{_BKG_FTP_PATH}/{year}/{doy:03d}/{fname}"
    print(f"[fetch_vtec] Trying BKG FTP: {_BKG_FTP_HOST}{remote_path}")
    try:
        with ftplib.FTP(_BKG_FTP_HOST, timeout=30) as ftp:
            ftp.login()   # anonymous
            with open(out_comp, "wb") as f:
                ftp.retrbinary(f"RETR {remote_path}", f.write)
        _decompress(out_comp, out_ion)
        out_comp.unlink(missing_ok=True)
        print(f"[fetch_vtec] Downloaded from BKG FTP → {out_ion.name}")
        return out_ion
    except Exception as e:
        print(f"[fetch_vtec] BKG FTP failed ({e})")
        if out_comp.exists():
            out_comp.unlink()
        return None


def _get_ionex(date: datetime.date, cache_dir: Path) -> Path:
    """
    Fetch IONEX for the given date, trying CDDIS then BKG.
    Raises RuntimeError if both fail.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ionex = _download_cddis(date, cache_dir)
    if ionex is None:
        ionex = _download_bkg(date, cache_dir)
    if ionex is None:
        raise RuntimeError(
            f"Could not download IONEX for {date}. "
            "Check your ~/.netrc credentials for CDDIS, or verify BKG FTP access."
        )
    return ionex


# ─── IONEX parsing (fallback if georinex not available) ───────────────────────

def _parse_ionex_simple(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Minimal IONEX parser that extracts TEC maps without georinex.
    Returns (lats, lons, times, maps) where maps is a list of 2D arrays [lat, lon].
    """
    lats = lons = times = None
    maps = []
    current_map = []
    in_map = False
    lat_vals = []

    with open(path) as f:
        for line in f:
            if "LAT1 / LAT2 / DLAT" in line:
                parts = line.split()
                lat1, lat2, dlat = float(parts[0]), float(parts[1]), float(parts[2])
                lats = np.arange(lat1, lat2 + dlat/2, dlat)
            elif "LON1 / LON2 / DLON" in line:
                parts = line.split()
                lon1, lon2, dlon = float(parts[0]), float(parts[1]), float(parts[2])
                lons = np.arange(lon1, lon2 + dlon/2, dlon)
            elif "START OF TEC MAP" in line:
                in_map = True
                current_map = []
                lat_vals = []
            elif "END OF TEC MAP" in line:
                in_map = False
                if current_map:
                    maps.append(np.array(current_map))  # [n_lat, n_lon]
            elif in_map and "LAT/LON1/LON2/DLON/H" in line:
                # IONEX uses fixed-width F8.1 fields; positive lat + negative lon
                # concatenate as e.g. "87.5-180.0" — regex is the safe parser.
                nums = re.findall(r'[-+]?\d+\.?\d*', line[:64])
                if nums:
                    lat_vals.append(float(nums[0]))
                    current_map.append([])
            elif in_map and lat_vals and "LAT/LON1/LON2/DLON/H" not in line \
                    and "END OF TEC MAP" not in line \
                    and "START OF TEC MAP" not in line \
                    and "EPOCH OF CURRENT MAP" not in line:
                vals = [float(v) for v in line.split() if v.lstrip("-").isdigit() or
                        (v.replace("-","").replace(".","").isdigit())]
                if vals and current_map:
                    current_map[-1].extend(vals)

    # Convert TEC units (0.1 TECU in IONEX → TECU)
    maps_arr = []
    for m in maps:
        try:
            arr = np.array(m, dtype=float)
            if arr.size > 0:
                maps_arr.append(arr * 0.1)
        except Exception:
            pass

    return lats, lons, maps_arr


# ─── Main API ─────────────────────────────────────────────────────────────────

def fetch_vtec_stats(lat: float,
                     lon: float,
                     date: str,
                     cache_dir: str = "./ionex_cache",
                     product: str = "igsg") -> dict:
    """Fetch IGS GIM IONEX for the given date and return VTEC statistics at
(lat, lon) across all available TEC maps in the file (typically 13 maps
at 2-hour intervals over 24 hours).
    """
    global _PRODUCT
    _PRODUCT = product

    date_obj = datetime.date.fromisoformat(date)
    ionex_path = _get_ionex(date_obj, Path(cache_dir))

    if _HAS_GEORINEX:
        # Use georinex for robust parsing
        ds = gr.load(str(ionex_path))
        tec = ds["tec"]   # DataArray [time, lat, lon] in TECU

        # Interpolate to requested lat/lon
        # IONEX lon convention: 0–360; convert if needed
        lon_interp = lon % 360
        vtec_series = tec.interp(
            lat=lat, lon=lon_interp, method="linear"
        ).values.ravel()
        vtec_series = vtec_series[np.isfinite(vtec_series)]

    else:
        # Fallback: minimal parser
        print("[fetch_vtec] georinex not available; using minimal parser")
        lats, lons, maps = _parse_ionex_simple(ionex_path)

        if lats is None or len(maps) == 0:
            raise RuntimeError("IONEX parsing failed — install georinex for robust support")

        # Find nearest grid point
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_interp = lon % 360
        lons_pos = lons % 360
        lon_idx = int(np.argmin(np.abs(lons_pos - lon_interp)))

        vtec_series = np.array([
            m[lat_idx, lon_idx] for m in maps
            if m.shape[0] > lat_idx and m.shape[1] > lon_idx
        ])
        vtec_series = vtec_series[vtec_series > 0]

    if len(vtec_series) == 0:
        raise RuntimeError("No valid VTEC values found at the requested location.")

    return {
        "vtec_mean_tecu" : float(np.mean(vtec_series)),
        "vtec_std_tecu"  : float(np.std(vtec_series, ddof=1) if len(vtec_series) > 1 else 0.0),
        "vtec_min_tecu"  : float(np.min(vtec_series)),
        "vtec_max_tecu"  : float(np.max(vtec_series)),
        "n_maps"         : int(len(vtec_series)),
        "date"           : date,
        "lat"            : lat,
        "lon"            : lon,
    }


def vtec_for_pipeline(lat: float,
                      lon: float,
                      date: str,
                      cache_dir: str = "./ionex_cache",
                      cv_floor: float = 0.3) -> dict:
    """Convenience wrapper: fetch VTEC stats and return a dict ready to merge
into sim_config.json.
    """
    stats = fetch_vtec_stats(lat, lon, date, cache_dir)
    mean  = stats["vtec_mean_tecu"]
    std   = max(stats["vtec_std_tecu"], cv_floor * mean)

    print(f"\n[fetch_vtec] VTEC at ({lat:.1f}°, {lon:.1f}°) on {date}")
    print(f"  Maps used        : {stats['n_maps']}")
    print(f"  Diurnal range    : {stats['vtec_min_tecu']:.1f} – "
          f"{stats['vtec_max_tecu']:.1f} TECU")
    print(f"  Mean (→ config)  : {mean:.2f} TECU")
    print(f"  Std  (→ config)  : {std:.2f} TECU  (CV={std/mean:.2f})")

    return {
        "VTEC_MEAN_TECU": round(mean, 2),
        "VTEC_STD_TECU" : round(std,  2),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Fetch IGS GIM VTEC and print pipeline-ready parameters"
    )
    parser.add_argument("--lat",       type=float, required=True,  help="Latitude [deg]")
    parser.add_argument("--lon",       type=float, required=True,  help="Longitude [deg]")
    parser.add_argument("--date",      type=str,   required=True,  help="Date YYYY-MM-DD")
    parser.add_argument("--cache_dir", type=str,   default="./ionex_cache")
    parser.add_argument("--product",   type=str,   default="igsg",
                        help="IGS product: igsg (combined), jplg (JPL)")
    args = parser.parse_args()

    cfg = vtec_for_pipeline(args.lat, args.lon, args.date, args.cache_dir)
    print("\nAdd to sim_config.json:")
    print(json.dumps(cfg, indent=2))
