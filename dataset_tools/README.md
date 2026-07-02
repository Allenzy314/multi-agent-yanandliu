# FLARE25 Ultrasound — Dataset 509 / 510 download & prepare

Tools to download the **raw public sources** for the Ultrasound datasets and
convert them into an **nnU-Net-style** layout.

> **Scope: 509 and 510 only.** Dataset **511 (Fetal_NT)** is excluded because its
> sole source, the FUSEP Kaggle dataset, was taken down and now returns HTTP 403.
> The 511 row below is kept for reference; the CLI no longer offers it.

## Dataset → source mapping

These were matched against `flare_dataset/MICCAI-FLARE25-AgentToolSet-Release.xlsx`
(sheet `datasets`) by train/test counts, class count and dataset name.

| Folder | Sources | datasets sheet row | Labels |
|---|---|---|---|
| `Dataset509_CAMUS_Left_Heart` | **CAMUS** (CREATIS) | row 15 (17264/1968, 3 cls) | LV, Myo, LA |
| `Dataset510_Four_Chamber_Heart` | **CardiacUDC + CardiacNet + EchoCP** | rows 16/17/18 (4503/565, 4 cls) | LV, LA, RA, RV |
| ~~`Dataset511_Fetal_NT`~~ (unavailable) | ~~FUSEP~~ — **removed from Kaggle (HTTP 403)** | row 13 (2678/477, 14 cls) | 14 fetal NT structures |

Source links:
- CardiacUDC — https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset
- CardiacNet — https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos
- EchoCP — https://www.kaggle.com/datasets/xiaoweixumedicalai/echocp
- ~~FUSEP — https://www.kaggle.com/datasets/liwenwang0919/fusep-datasets~~ (HTTP 403 — removed)
- CAMUS — https://www.creatis.insa-lyon.fr/Challenge/camus/ (free account required)

## Conversion fidelity

Both 509 and 510 are **faithful**: the converter reproduces the exact official
filenames and writes straight into the real `imagesTr/labelsTr`, so the existing
`dataset.json` is **directly usable** (verified by 100% coverage checks).

- **509 CAMUS.** Slices `*_half_sequence` + `*_half_sequence_gt` into
  `patientXXXX_{2CH,4CH}_tNN` — **17264/17264** training frames present; held-out
  test patients go to `imagesTs/labelsTs` (1968). Labels map 1=LV/2=Myo/3=LA.
- **510 Four Chamber.** Maps each `dataset.json` name back to its raw source
  (EchoCP / CardiacUDC / CardiacNet) and slices the named frame —
  **4503/4503** training frames present, labels 1=LV/2=LA/3=RA/4=RV. Two CardiacNet
  label files are truncated in the Kaggle upload; their 14 frames are recovered by
  a partial NIfTI read, so all 4503 labels are produced.

### Test sets
- **509** `imagesTs/labelsTs` = 1968 (the 50 patients not in the training split).
- **510** `--with-test` builds `imagesTs/labelsTs` from every labeled frame whose
  source **volume contributes no training frame** — a **leakage-free** held-out set
  (≈696 frames). Note: this is a *superset* of the organizers' official 565-frame
  test split, which **cannot** be reconstructed from the public json (they
  subsampled, and the test names aren't listed). Frames left over from *training*
  volumes are deliberately excluded to avoid train/test contamination.

Orientation follows the raw NIfTI (image & label are always co-aligned); intensities
are the original 0–255. The exact pixel orientation may differ from the organizers'
PNGs by a transpose/flip, but that does not affect training (pairs stay aligned).

## Setup

```bash
pip install -r dataset_tools/requirements.txt
# Kaggle token at ~/.kaggle/kaggle.json  (chmod 600)
chmod 600 ~/.kaggle/kaggle.json

# 7-Zip — REQUIRED to extract the CardiacUDC / EchoCP spanned archives (see below).
conda install -c conda-forge p7zip        # no sudo (provides `7za`)
# or:  sudo apt-get install -y p7zip-full  # provides `7z`
```

## 1) Download

```bash
# 509 + 510 (CAMUS will print manual instructions unless you pass a URL/mirror)
python dataset_tools/download_datasets.py

# just the Kaggle-backed one
python dataset_tools/download_datasets.py --datasets 510

# automate CAMUS if you have a direct zip or a Kaggle mirror slug
python dataset_tools/download_datasets.py --datasets 509 --camus-url  "<ZIP_URL>"
python dataset_tools/download_datasets.py --datasets 509 --camus-kaggle "<user/slug>"
```

Downloads land in `DatasetXXX/raw_source/<SourceName>/` and are idempotent
(existing folders are skipped).

## 2) Inspect what was downloaded

```bash
python dataset_tools/convert_to_nnunet.py --inspect
```

This prints the real file inventory (extensions + counts + an example path).
Use it to confirm the raw structure before converting — and share it if the
converter needs source-specific tweaks.

## 3) Convert to nnU-Net layout

```bash
# train sets for both, plus 510's held-out test set:
python dataset_tools/convert_to_nnunet.py --datasets 509 510 --with-test

# 509 CAMUS only (reads CAMUS_public.zip directly — no extraction needed):
python dataset_tools/convert_to_nnunet.py --datasets 509
# 510 train only, or with the leakage-free held-out test set:
python dataset_tools/convert_to_nnunet.py --datasets 510
python dataset_tools/convert_to_nnunet.py --datasets 510 --with-test
# quick smoke test:
python dataset_tools/convert_to_nnunet.py --datasets 510 --limit 20
```

Output (both faithful — official names, `dataset.json` works as-is):
```
Dataset509_CAMUS_Left_Heart/
  imagesTr/patientXXXX_2CH_tNN_0000.png
  labelsTr/patientXXXX_2CH_tNN.png            # 0=bg,1=LV,2=Myo,3=LA   (17264)
  imagesTs/ , labelsTs/                       # 50 held-out test patients (1968)

Dataset510_Four_Chamber_Heart/
  imagesTr/<official-name>_0000.png           # names from EchoCP/CardiacUDC/CardiacNet
  labelsTr/<official-name>.png                # 0=bg,1=LV,2=LA,3=RA,4=RV (4503)
```

### What the converter does automatically
- **509 CAMUS** (dedicated path): reads `CAMUS_public.zip` in place, slices each
  `*_half_sequence` + `*_half_sequence_gt` into the official `patientXXXX_{view}_tNN`
  frames, routing training vs. test patients per `dataset.json`.
- **510 Four Chamber** (dedicated path): for every `dataset.json` name, resolves the
  raw source by pattern — EchoCP `{id}_{v|r}_{frame}`, CardiacUDC
  `Site_X_NN_{stem}_{frame}` / `label_all_frame_{stem}_{frame}`, CardiacNet
  `{subgroup}_{id}_{stem}_{frame}` (image entry is a `…_image.nii/` **directory**) —
  then slices that frame from the matching `*_image`/`*_label` NIfTI. Remaps the lone
  CardiacUDC `normal-27-4` label value 6→4 and partial-reads the 2 truncated
  CardiacNet label files.
- **Other datasets** fall back to the generic auto-converter (NIfTI slicing, video
  frame extraction, image+mask pairing → `*_generated/`).
- **511 Fetal NT** *(unavailable — FUSEP removed from Kaggle, HTTP 403)*: not
  downloadable and excluded from the CLI. If you locate a working mirror, drop the
  `"unavailable"` flag in `config.py` to re-enable it. It is a **detection** dataset
  (boxes), so use `dataset_detect.json` rather than segmentation masks.

## Troubleshooting — Dataset 510 "a link won't download"

The three 510 sources are hosted **only on Kaggle** (their GitHub repos just point
back to Kaggle). The failures are about *archive format / size*, not dead links:

| Source | Kaggle slug | Format | Gotcha |
|---|---|---|---|
| **CardiacUDC** | `xiaoweixumedicalai/cardiacudc-dataset` | **7-part spanned zip** (`.change2zip` + `.z01`–`.z06`, ~4.5 GB) | needs 7-Zip reassembly |
| **EchoCP** | `xiaoweixumedicalai/echocp` | **2-part spanned zip** (`.change2zip` + `.z01`, ~5.5 GB) + xlsx | needs 7-Zip reassembly |
| **CardiacNet** | `xiaoweixumedicalai/abnormcardiacechovideos` | plain `.nii`, **~70 GB** | huge; downloads time out, not broken |

**Why it looks broken:** CardiacUDC and EchoCP are uploaded as an Info-ZIP *split*
archive whose final volume was renamed `.zip` → `.change2zip` (so Kaggle accepts it).
A plain `kaggle download + unzip` can't open `.change2zip`, and missing any one
`.z0x` part makes the whole set unextractable — the classic "won't download/open".

**Fix (built into the downloader):** after download it auto-detects the
`.change2zip` set and extracts it with **7-Zip**. If 7-Zip is missing it prints an
install hint and continues; install it (see Setup) and re-run just the extraction:

```bash
python dataset_tools/download_datasets.py --datasets 510 --reassemble-only
```

> ⚠️ Do **not** use `zip -s 0 X.zip --out Y.zip` to recombine — verified to corrupt
> the data on this machine's Info-ZIP 3.0 (`unzip -t` → "invalid compressed data").
> 7-Zip is the only reliable extractor here.

### Verified alternative / canonical links (from a source-triage sweep)
- **CardiacUDC** — pointer repos (data still on Kaggle):
  https://github.com/XiaoweiXu/CardiacUDA-dataset ·
  https://github.com/xmed-lab/GraphEcho (ICCV'23, arXiv:2309.11145)
- **CardiacNet** — alternative data hosts (avoid Kaggle's 70 GB zip):
  https://github.com/xmed-lab/CardiacNet ·
  OneDrive (linked from that repo; arXiv:2410.20769).
  Note: `github.com/XiaoweiXu/CardiacNet-dataset` is **404** (dead).
- **EchoCP** — official mirror hosts the data:
  https://github.com/XiaoweiXu/EchoCP-An-Echocardiography-Dataset-in-Contrast-Transthoracic-Echocardiography-for-PFO-diagnosis
  (MICCAI'21, arXiv:2105.08267) — use this if the Kaggle spanned zip is a problem.

### CardiacNet structure quirk (handled)
Each image entry is a **directory** named like `001_image.nii/` that *contains* the
real file (`001_image.nii/patient-2-4_image.nii`), while labels are plain
`001_label.nii` files. The 510 converter handles this directly (it resolves the
inner file from the `dataset.json` patient stem), so no manual step is needed.
Two `_label.nii` files (`Non-ASD/077`, `Non-PAH/089`) are **truncated** in the
Kaggle upload; the converter recovers their 14 referenced frames via a partial read.

## Integrity check (all datasets)

`check_integrity.py` validates every `DatasetXXX_*` folder under the Ultrasound
root against its JSON manifests and what is on disk:

```bash
python dataset_tools/check_integrity.py                 # all datasets, fast
python dataset_tools/check_integrity.py --datasets 509 510
python dataset_tools/check_integrity.py --deep          # + open PNGs, check label values (sampled)
python dataset_tools/check_integrity.py --deep-full     # deep check on every file (slow)
```

What it verifies:
- **dataset.json** (seg): every training `{image,label}` and every test image
  (the test split is a list of image-path strings) exists; counts match
  `numTraining`/`numTest`; no duplicate refs.
- **dataset_detect.json / dataset_cls.json**: every referenced image exists
  (resolving the `_0000` channel suffix that detect/cls names omit); box class
  ids and `cls` labels are within the declared label set.
- **disk pairing**: `imagesTr`↔`labelsTr` and `imagesTs`↔`labelsTs` line up;
  reports orphan labels / images and count mismatches.
- `--deep` also opens PNGs (catch corruption) and checks label pixel values are
  within the declared labels.

Severities: **ERROR** (missing referenced file, unreadable/out-of-range pixels,
unpopulated dataset), **WARN** (orphans, count mismatch, missing labelsTs),
**INFO** (leftover raw archives). Exit code is non-zero if any ERROR is found.
