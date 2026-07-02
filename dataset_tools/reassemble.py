#!/usr/bin/env python3
"""Reassemble + extract the renamed multi-volume ("spanned") zip archives that
some of the Kaggle sources ship (CardiacUDC, EchoCP).

Background
----------
Those datasets are uploaded as an Info-ZIP *split* archive whose final volume
was renamed `.zip` -> `.change2zip` (so Kaggle's uploader would accept it):

    cardiacUDC_dataset.change2zip   <- FINAL volume (holds the central directory)
    cardiacUDC_dataset.z01 .. .z06  <- earlier volumes

A plain `kaggle download + unzip` therefore yields no usable data: `unzip`
cannot open `.change2zip`, and Info-ZIP `unzip` has no true multi-volume support.

Reliable extraction needs **7-Zip** (`7z` / `7za` / `7zz`), which opens the
`.zip` final volume and automatically pulls in the `.z01..` parts.

IMPORTANT: do NOT use `zip -s 0 X.zip --out Y.zip` on Info-ZIP 3.0 here — it
recombines to a corrupt archive (verified: `unzip -t` fails with
"invalid compressed data"). 7-Zip is required.

If 7-Zip is missing, install it without sudo (you have miniconda):
    conda install -c conda-forge p7zip      # provides `7za`
or system-wide:
    sudo apt-get install -y p7zip-full      # provides `7z`
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess


def _log(msg: str) -> None:
    print(f"[reassemble] {msg}", flush=True)


def find_7z() -> str | None:
    """Return a usable 7-Zip executable path, or None."""
    # Common names: 7z (p7zip-full), 7za (p7zip / conda), 7zz (7zip conda-forge)
    for name in ("7z", "7zz", "7za"):
        p = shutil.which(name)
        if p:
            return p
    # Fall back to common conda install locations (base env + conda envs),
    # without recursively scanning the whole home directory.
    bins = []
    for root in (os.path.expanduser("~/miniconda3"), os.path.expanduser("~/anaconda3"),
                 os.environ.get("CONDA_PREFIX", "")):
        if not root:
            continue
        bins.append(os.path.join(root, "bin"))
        bins.extend(glob.glob(os.path.join(root, "envs", "*", "bin")))
    for b in bins:
        for name in ("7z", "7zz", "7za"):
            cand = os.path.join(b, name)
            if os.access(cand, os.X_OK):
                return cand
    return None


def _normalize_change2zip(folder: str) -> list[str]:
    """Rename '<base>.change2zip' (and a Kaggle-appended '<base>.change2zip.zip')
    to '<base>.zip'. Returns the list of resulting main '.zip' volume paths."""
    mains: list[str] = []
    # Kaggle sometimes saves a single-file download as '<name>.change2zip.zip'
    for p in glob.glob(os.path.join(folder, "**", "*.change2zip.zip"), recursive=True):
        fixed = p[: -len(".change2zip.zip")] + ".change2zip"
        os.replace(p, fixed)
        _log(f"  unwrapped Kaggle suffix: {os.path.basename(p)} -> {os.path.basename(fixed)}")
    for p in glob.glob(os.path.join(folder, "**", "*.change2zip"), recursive=True):
        main = p[: -len(".change2zip")] + ".zip"
        if not os.path.exists(main):
            os.replace(p, main)
            _log(f"  renamed {os.path.basename(p)} -> {os.path.basename(main)}")
        mains.append(main)
    return mains


def _parts_present(main_zip: str) -> list[str]:
    base = main_zip[: -len(".zip")]
    parts = sorted(glob.glob(base + ".z[0-9][0-9]"))
    return parts


def reassemble_folder(folder: str) -> bool:
    """Find any '*.change2zip' spanned archive under `folder`, extract it with
    7-Zip in place. Returns True if work was done successfully, False if there
    was nothing to do. Raises RuntimeError on a real failure (e.g. no 7-Zip).

    Also picks up an already-renamed split set (a '.zip' with sibling '.z01'
    parts), so re-running after a failed attempt still extracts it."""
    if not os.path.isdir(folder):
        return False

    mains = _normalize_change2zip(folder)
    # Also include '<base>.zip' files that have a sibling '.z01' part — i.e. a
    # split set whose '.change2zip' was already renamed on a previous run.
    for z in glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True):
        if z not in mains and os.path.exists(z[:-len(".zip")] + ".z01"):
            mains.append(z)
    if not mains:
        return False

    sevenzip = find_7z()
    if not sevenzip:
        raise RuntimeError(
            "Found a multi-volume '.change2zip' archive but no 7-Zip executable.\n"
            "  Install without sudo (you have conda):  conda install -c conda-forge p7zip\n"
            "  or system-wide:                          sudo apt-get install -y p7zip-full\n"
            "  Then re-run:  python dataset_tools/download_datasets.py --reassemble-only\n"
            "  (Do NOT use `zip -s 0` — it corrupts the archive on Info-ZIP 3.0.)"
        )

    ok_any = False
    for main in mains:
        parts = _parts_present(main)
        out_dir = os.path.dirname(main)
        _log(f"  extracting {os.path.basename(main)} "
             f"(+{len(parts)} part(s)) with {os.path.basename(sevenzip)} -> {out_dir}")
        if parts:
            expected = [f"{main[:-4]}.z{ i:02d}" for i in range(1, len(parts) + 1)]
            missing = [e for e in expected if not os.path.exists(e)]
            if missing:
                raise RuntimeError(
                    f"Spanned archive {os.path.basename(main)} is missing parts: "
                    f"{', '.join(os.path.basename(m) for m in missing)}. "
                    "Re-download the dataset completely — a partial download cannot extract."
                )
        # 7-Zip auto-discovers .z01.. when pointed at the final .zip volume.
        res = subprocess.run(
            [sevenzip, "x", "-y", f"-o{out_dir}", main],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"7-Zip failed on {os.path.basename(main)} (rc={res.returncode}):\n"
                + res.stdout[-1500:]
            )
        ok_any = True
        _log(f"  OK: extracted {os.path.basename(main)}")
    return ok_any


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract *.change2zip spanned archives in a folder")
    ap.add_argument("folder", help="folder containing the .change2zip / .z01.. parts")
    args = ap.parse_args()
    did = reassemble_folder(args.folder)
    _log("done" if did else "nothing to reassemble")
