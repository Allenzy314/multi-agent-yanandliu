"""Shared configuration for the FLARE25 Ultrasound dataset tools.

Maps the three nnU-Net dataset folders (509 / 510 / 511) to their original
public data sources, as recorded in
`flare_dataset/MICCAI-FLARE25-AgentToolSet-Release.xlsx` (sheet `datasets`).

Source -> dataset mapping
-------------------------
  Dataset509_CAMUS_Left_Heart    -> CAMUS                (datasets sheet row 15)
  Dataset510_Four_Chamber_Heart  -> CardiacUDC + CardiacNet + EchoCP (rows 16/17/18)
  Dataset511_Fetal_NT            -> FUSEP                (datasets sheet row 13)
"""

from __future__ import annotations

import os

# Default data root = the Ultrasound folder that already holds the dataset.json
# files. `flare_dataset` is a symlink to the real dataset location, which is fine.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "flare_dataset", "Ultrasound")

# Name of the subfolder created inside each DatasetXXX/ to hold raw downloads.
RAW_SUBDIR = "raw_source"

SOURCES = {
    "509": {
        "folder": "Dataset509_CAMUS_Left_Heart",
        "labels": {"background": 0, "LV": 1, "Myo": 2, "LA": 3},
        "kaggle": [],  # CAMUS is not on Kaggle officially; see `camus` below.
        # Official CAMUS challenge (CREATIS / humanheart-project). Requires a free
        # account, so it cannot be fully automated. Provide a direct zip URL via
        # --camus-url, or a Kaggle mirror slug via --camus-kaggle, to automate it.
        "camus": {
            "homepage": "https://www.creatis.insa-lyon.fr/Challenge/camus/",
            "download_page": "https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8",
            # nnU-Net labels for this dataset (from dataset.json):
            "labels": {"background": 0, "LV": 1, "Myo": 2, "LA": 3},
        },
    },
    "510": {
        "folder": "Dataset510_Four_Chamber_Heart",
        # (subfolder name, kaggle dataset slug)
        "kaggle": [
            ("CardiacUDC", "xiaoweixumedicalai/cardiacudc-dataset"),
            ("CardiacNet", "xiaoweixumedicalai/abnormcardiacechovideos"),
            ("EchoCP",     "xiaoweixumedicalai/echocp"),
        ],
        "labels": {"background": 0, "LV": 1, "LA": 2, "RA": 3, "RV": 4},
    },
    "511": {
        "folder": "Dataset511_Fetal_NT",
        # UNAVAILABLE: the FUSEP Kaggle dataset was taken down / made private
        # (the slug below now returns HTTP 403 Forbidden), so 511 cannot be
        # downloaded and is excluded from the CLI. Kept here only for reference.
        "unavailable": "FUSEP Kaggle dataset removed (HTTP 403 Forbidden)",
        "kaggle": [
            ("FUSEP", "liwenwang0919/fusep-datasets"),
        ],
        # 14-class detection dataset; segmentation labels are sparse. See README.
        "labels": {
            "Maxilla": 0, "Mandible": 1, "LV": 2, "Head": 3, "RBP": 4, "DP": 5,
            "thalami": 6, "plate": 7, "IT": 8, "CM": 9, "midbrain": 10,
            "NT": 11, "NTAPS": 12, "NB": 13,
        },
    },
}

# Datasets that can actually be fetched/processed. 511 is intentionally excluded
# because its only source (FUSEP) was removed from Kaggle (HTTP 403).
AVAILABLE_KEYS = [k for k, v in SOURCES.items() if "unavailable" not in v]


def dataset_dir(data_root: str, key: str) -> str:
    """Absolute path to the DatasetXXX_... folder for key in {509,510,511}."""
    return os.path.join(data_root, SOURCES[key]["folder"])


def raw_dir(data_root: str, key: str) -> str:
    """Absolute path to the raw download folder for a dataset."""
    return os.path.join(dataset_dir(data_root, key), RAW_SUBDIR)
