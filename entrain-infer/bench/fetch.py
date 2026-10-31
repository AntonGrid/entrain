"""Fetch the two real backgrounds and write ``bench/manifest.json``.

The bench never invents data. If a download is missing the bench refuses to run
rather than quietly substituting a model -- an unmeasured background is worse
than no background, because it looks like one.

Both sources are open and need no key:

* NASA POWER, hourly ``ALLSKY_SFC_SW_DWN`` (W/m^2), 2021, point 56.13 N 40.40 E;
* PVGIS (JRC), hourly global irradiance ``G(i)`` (W/m^2), 2020, same point.

Usage::

    python bench/fetch.py            # download what is missing, write manifest
    python bench/fetch.py --check    # verify hashes of what is already there
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).resolve().parent
DATA = BENCH / "data"
MANIFEST = BENCH / "manifest.json"

SOURCES: dict[str, dict] = {
    "nasa_power_2021": {
        "filename": "nasa_power_2021_vladimir.json",
        "family": "NASA POWER (NASA Langley) -- open, no key",
        "description": "Hourly all-sky shortwave irradiance ALLSKY_SFC_SW_DWN, W/m^2",
        "place": "56.13 N, 40.40 E (Vladimir region)",
        "period": "2021-01-01 .. 2021-12-31",
        "expected_samples": 8760,
        "url": "https://power.larc.nasa.gov/api/temporal/hourly/point",
        "query": {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": "40.40",
            "latitude": "56.13",
            "start": "20210101",
            "end": "20211231",
            "format": "JSON",
        },
    },
    "pvgis_2020": {
        "filename": "pvgis_2020_vladimir.json",
        "family": "PVGIS (European Commission JRC) -- open, no key",
        "description": "Hourly global irradiance on the plane G(i), W/m^2",
        "place": "56.13 N, 40.40 E (Vladimir region)",
        "period": "2020-01-01 .. 2020-12-31",
        "expected_samples": 8784,
        "url": "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc",
        "query": {
            "lat": "56.13",
            "lon": "40.40",
            "startyear": "2020",
            "endyear": "2020",
            "horirrad": "1",
            "outputformat": "json",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, force: bool = False) -> Path:
    """Download one source into ``bench/data`` unless it is already there."""
    spec = SOURCES[name]
    path = DATA / spec["filename"]
    if path.exists() and not force:
        return path
    DATA.mkdir(parents=True, exist_ok=True)
    url = spec["url"] + "?" + urllib.parse.urlencode(spec["query"])
    print(f"[fetch] {name} <- {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def summarise(name: str, path: Path) -> dict:
    spec = dict(SOURCES[name])
    spec["path"] = str(path.relative_to(BENCH.parent))
    spec["bytes"] = path.stat().st_size
    spec["sha256"] = sha256(path)
    spec["fetched_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return spec


def write_manifest() -> Path:
    entries = {}
    for name in SOURCES:
        path = download(name)
        entries[name] = summarise(name, path)
    manifest = {
        "note": "sha256 of every background the bench reads; regenerate with fetch.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return MANIFEST


def check_manifest() -> int:
    """Fail loudly if a byte of the recorded background has moved."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for name, entry in manifest["sources"].items():
        path = BENCH.parent / entry["path"]
        if not path.exists():
            print(f"[check] MISSING {name}: {path}")
            bad += 1
            continue
        actual = sha256(path)
        ok = actual == entry["sha256"]
        print(f"[check] {name}: {'ok' if ok else 'HASH MISMATCH'} {actual[:16]}")
        bad += int(not ok)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify hashes only")
    parser.add_argument("--force", action="store_true", help="re-download everything")
    args = parser.parse_args(argv)
    if args.check:
        return check_manifest()
    for name in SOURCES:
        download(name, force=args.force)
    print(f"[fetch] manifest -> {write_manifest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
