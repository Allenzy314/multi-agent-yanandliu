#!/usr/bin/env python3
"""Download the raw source data for FLARE25 Ultrasound Datasets 509 and 510.

(511 / Fetal_NT is excluded: its only source — the FUSEP Kaggle dataset — was
removed and now returns HTTP 403, so it cannot be downloaded.)

Each dataset's original sources (see config.SOURCES) are downloaded into
    <data_root>/DatasetXXX_.../raw_source/<SourceName>/

The existing dataset.json / dataset_detect.json files are never touched.

Requirements
------------
  pip install kaggle
  Kaggle API token at ~/.kaggle/kaggle.json  (chmod 600), or KAGGLE_CONFIG_DIR set.

Usage
-----
  python dataset_tools/download_datasets.py                 # 509 + 510
  python dataset_tools/download_datasets.py --datasets 510
  python dataset_tools/download_datasets.py --datasets 509 --camus-url <ZIP_URL>
  python dataset_tools/download_datasets.py --datasets 509 --camus-kaggle <user/slug>
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

from config import AVAILABLE_KEYS, DEFAULT_DATA_ROOT, SOURCES, dataset_dir, raw_dir
from reassemble import reassemble_folder


def _log(msg: str) -> None:
    print(f"[download] {msg}", flush=True)


def _maybe_reassemble(dest: str) -> None:
    """Extract any renamed multi-volume '.change2zip' archive in dest.
    A missing 7-Zip is reported as a NOTE (not fatal) so the run continues."""
    try:
        if reassemble_folder(dest):
            _log(f"  reassembled spanned archive under {dest}")
    except RuntimeError as exc:  # e.g. 7-Zip missing or a part is missing
        _log(f"  NOTE: {exc}")


def _ensure_kaggle():
    """Authenticate the Kaggle API, with a clear error if not configured."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("kaggle package not installed. Run: pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "Kaggle authentication failed. Put your token at ~/.kaggle/kaggle.json "
            f"(chmod 600) or set KAGGLE_CONFIG_DIR.\nUnderlying error: {exc}"
        )
    return api


def _already_populated(path: str) -> bool:
    return os.path.isdir(path) and any(os.scandir(path))


def download_kaggle(api, slug: str, dest: str) -> None:
    """Download + unzip a Kaggle dataset into dest (idempotent), then extract
    any multi-volume '.change2zip' spanned archive it contains."""
    if _already_populated(dest):
        _log(f"  skip download (already present): {dest}")
    else:
        os.makedirs(dest, exist_ok=True)
        _log(f"  kaggle: {slug} -> {dest}")
        api.dataset_download_files(slug, path=dest, unzip=True, quiet=False)
    _maybe_reassemble(dest)


def download_camus(cfg: dict, dest: str, camus_url: str | None,
                   camus_kaggle: str | None, api) -> None:
    """CAMUS: automate via a direct zip URL or a Kaggle mirror, else instruct."""
    if _already_populated(dest):
        _log(f"  skip (already present): {dest}")
        return
    os.makedirs(dest, exist_ok=True)

    if camus_kaggle:
        if api is None:
            api = _ensure_kaggle()
        _log(f"  CAMUS via Kaggle mirror: {camus_kaggle}")
        api.dataset_download_files(camus_kaggle, path=dest, unzip=True, quiet=False)
        return

    if camus_url:
        zip_path = os.path.join(dest, "camus.zip")
        _log(f"  CAMUS via direct URL -> {zip_path}")
        urllib.request.urlretrieve(camus_url, zip_path)
        _log("  extracting CAMUS zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        os.remove(zip_path)
        return

    cam = cfg["camus"]
    _log("  CAMUS needs a free account and cannot be auto-downloaded by default.")
    _log(f"    1) Register / open: {cam['homepage']}")
    _log(f"    2) Download the full dataset from: {cam['download_page']}")
    _log(f"    3) Unzip it into: {dest}")
    _log("    Or re-run with --camus-url <zip_url>  /  --camus-kaggle <user/slug>")


def run(data_root: str, keys: list[str], camus_url: str | None,
        camus_kaggle: str | None, reassemble_only: bool = False,
        sources: list[str] | None = None) -> None:
    api = None  # lazily authenticate only if a Kaggle source is requested
    want = {s.lower() for s in sources} if sources else None
    for key in keys:
        cfg = SOURCES[key]
        _log(f"=== {cfg['folder']} ===")
        if not os.path.isdir(dataset_dir(data_root, key)):
            _log(f"  WARNING: dataset folder missing: {dataset_dir(data_root, key)}")
        base_raw = raw_dir(data_root, key)
        os.makedirs(base_raw, exist_ok=True)

        if reassemble_only:
            # Skip downloading; one recursive pass extracts any spanned archive
            # under raw_source/ (covers every source subfolder).
            _maybe_reassemble(base_raw)
            continue

        for source_name, slug in cfg.get("kaggle", []):
            if want is not None and source_name.lower() not in want:
                _log(f"  skip (not in --sources): {source_name}")
                continue
            if api is None:
                api = _ensure_kaggle()
            download_kaggle(api, slug, os.path.join(base_raw, source_name))

        if "camus" in cfg:
            download_camus(cfg, os.path.join(base_raw, "CAMUS"),
                           camus_url, camus_kaggle, api)

    _log("Done. Inspect with: python dataset_tools/convert_to_nnunet.py --inspect")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help=f"Ultrasound folder (default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--datasets", nargs="+", default=AVAILABLE_KEYS,
                    choices=AVAILABLE_KEYS,
                    help="Which datasets to fetch (511 excluded: FUSEP removed from Kaggle)")
    ap.add_argument("--camus-url", default=None,
                    help="Direct .zip URL for CAMUS (automates Dataset509)")
    ap.add_argument("--camus-kaggle", default=None,
                    help="Kaggle slug (user/dataset) for a CAMUS mirror")
    ap.add_argument("--reassemble-only", action="store_true",
                    help="Skip downloading; only extract already-downloaded "
                         "'.change2zip' spanned archives (needs 7-Zip)")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="Only fetch these named sources, e.g. "
                         "--sources CardiacUDC EchoCP (default: all)")
    args = ap.parse_args()
    run(args.data_root, args.datasets, args.camus_url, args.camus_kaggle,
        reassemble_only=args.reassemble_only, sources=args.sources)


if __name__ == "__main__":
    main()
