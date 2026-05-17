"""Build Kaggle master notebooks: 08a + 08b, targeting Solar Energy (Elsevier).

Two-notebook split for Kaggle's 12-hour session limit. Same total content as
the Colab single-notebook version. Output: one downloadable zip from 08b.

  08a_master_part1_kaggle.ipynb (~9-10h):  data + VAE + main SolarSDE + 5-fold CV
  08b_master_part2_kaggle.ipynb (~6-8h):   baselines + ablations + stats + figures + tables

Target journal: Solar Energy (Elsevier, IF ~6, hybrid — free traditional publishing).
"""

import json, sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from _combined_generator import (
    build_nb, SHARED_CODE,
    STAGE_MINUS1_CODE, STAGE0_CODE,
    BASELINES_CODE, ABLATIONS_CODE,
    CALIBRATION_CODE, STRATIFIED_CODE, ANALYSIS_CODE,
)
from _generator import (
    BMS_DOWNLOAD, PREPROCESS_CODE,
    VAE_MODEL, IMAGE_DATASET, VAE_TRAIN, LATENT_EXTRACT,
)
from _final_generator import (
    LOAD_DATA_TOLERANT_CODE, CORRECTED_INFERENCE_CODE,
    RAMP_AUROC_CODE, BOOTSTRAP_CIS_CODE, PIT_RELIABILITY_CODE,
    ECONOMIC_CAISO_CODE, LATEX_TABLES_CODE, EXTRA_ABLATIONS_CODE,
    GOLDEN_RETRAIN_GUARD_CODE, GOLDEN_KT_PHYS_CODE, GOLDEN_EXTENDED_CODE,
    CLOUDCV_DOWNLOAD_ROBUST, CLOUDCV_EXTRACT_ROBUST,
    ZIP_DOWNLOAD_CODE, _gate,
)
from _colab_master_generator import (
    K_FOLD_CV_CODE, CTI_VALIDATION_CODE, HOLM_BONFERRONI_CODE,
)
from _master_hardening import (
    PREFLIGHT_SANITY_CODE, POST_STAGE0_VERIFY_CODE,
    STAGE_M1_SAFE_FALLBACK_CODE, safe_stage,
)
from _solarsde_v2 import (
    MDN_ARCHITECTURE_CODE, STAGE_0_V2_CODE, POST_STAGE0_V2_VERIFY_CODE,
)


HEADER_08A_MD = """# SolarSDE Master Part 1 — Foundations + Training + Cross-Validation

**Target venue: Solar Energy (Elsevier, IF ~6, hybrid — no mandatory APC).**

**Kaggle workflow:** Run this notebook (~9-10h on P100), then save the version
as a Kaggle Dataset. Open 08b_master_part2_kaggle.ipynb and attach this run's
output as an input dataset. ~6-8h more. Final results zip downloads from 08b.

## What this notebook does

| Step | What | Time |
|------|------|------|
| Setup + soft fast-start | Pull cached Golden artifacts from GitHub if available | 5 min |
| Retrain Golden (conditional) | CloudCV download (~2.6 GB) + BMS + preprocess + VAE training (20 epochs) + latent extraction + physics features + extended splits | ~3-4h |
| Image features | Optical flow + sun-ROI + cloud fraction on Golden test/val/train | ~30 min |
| Train SolarSDE | CTI-gated Neural SDE + Score Decoder, seed 42 | ~3-4h |
| 5-fold leave-one-day-out CV | Reviewer-required validation across train days | ~2.5h |
| Zip outputs | Single zip + summary CSV in /kaggle/working/ | <5 min |

## Kaggle setup

1. Settings → Accelerator: **GPU P100** (or T4 if P100 unavailable)
2. Settings → Internet: **On**
3. Run all cells
4. When done: File → Save Version → Save & Run All (creates a Kaggle Dataset
   from /kaggle/working/)
5. Download `solarsde_outputs.zip` from the Output tab
6. Open 08b_master_part2_kaggle.ipynb. Add Data → Your Datasets → attach the
   version you just saved. Run 08b.
"""


HEADER_08B_MD = """# SolarSDE Master Part 2 — Baselines, Ablations, Statistical Pack, Figures

**Run 08a_master_part1_kaggle.ipynb FIRST.** Attach its output as a Kaggle
Dataset before running this notebook.

**Target venue: Solar Energy (Elsevier).**

## What this notebook does

| Step | What | Time |
|------|------|------|
| Setup + dataset attach | Copy Part 1 artifacts into /kaggle/working/ | 1 min |
| Soft fast-start | Pull anything missing from GitHub cache | 1-5 min |
| Stage C+ corrected inference | Future-covariate advancement + per-horizon prediction saves | ~30 min |
| 5 standard baselines | Persistence, Smart-Pers, LSTM, MC-Dropout, CSDI | ~2-3h |
| 4 ablations | A2 (no-CTI), A3 (no-VAE PCA), A4 (no-score), A5 (no-SDE/ODE) | ~2h |
| Conformal calibration | Post-hoc calibration to fix PICP coverage | 10 min |
| Stratified eval + DM test | By CTI quartile / regime / ramp / zenith / time-of-day | 30 min |
| PIT + reliability + bootstrap CIs | Standard probabilistic-forecast diagnostics | 30 min |
| Ramp AUROC + CTI lead-time | Operational ramp detection figure | 10 min |
| CTI vs cloud-cover validation | Spearman correlation (physical-meaningfulness for Solar Energy reviewers) | 10 min |
| Holm-Bonferroni correction | Multiple-comparison correction on DM p-values | <5 min |
| Economic value | Resource-cost simulation (USD/yr per GW) | 15 min |
| Analysis figures | CTI dynamics + regime analysis + forecast traces | 15 min |
| LaTeX tables | 3 paper-ready tables | <5 min |
| Final zip | Everything bundled to /kaggle/working/ for download | 5 min |
"""


KAGGLE_SETUP_PART1_CODE = '''\
# ==== Kaggle setup (Part 1) ====
import os, sys, shutil
from pathlib import Path

IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
IN_COLAB = "google.colab" in sys.modules
assert IN_KAGGLE or not IN_COLAB, "This notebook is designed for Kaggle. Use 08_solarsde_master_colab.ipynb on Colab."

PERSIST_DIR = Path("/kaggle/working/solarsde_outputs") if IN_KAGGLE else (Path.cwd() / "solarsde_outputs")
WORK_DIR    = Path("/kaggle/working/solarsde") if IN_KAGGLE else (Path.cwd() / "solarsde_work")

for d in [PERSIST_DIR, WORK_DIR,
          PERSIST_DIR / "checkpoints", PERSIST_DIR / "results",
          PERSIST_DIR / "latents",     PERSIST_DIR / "splits",
          PERSIST_DIR / "extended",    PERSIST_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# If you attached your previous solarsde_outputs folder as a Kaggle dataset
# (e.g., from a 07a1 run), auto-copy it in so the retrain stage skips.
if IN_KAGGLE:
    src_root = Path("/kaggle/input")
    if src_root.exists():
        print("Checking attached Kaggle datasets for cached artifacts ...")
        for ds in src_root.iterdir():
            if not ds.is_dir(): continue
            looks_cached = ((ds / "checkpoints").exists()
                            or (ds / "splits").exists()
                            or (ds / "latents").exists())
            if not looks_cached: continue
            print(f"  found candidate: {ds.name}")
            for sub in ds.iterdir():
                if not sub.is_dir(): continue
                dst = PERSIST_DIR / sub.name
                if dst.exists() and any(dst.iterdir()):
                    print(f"    skip {sub.name}/ (already populated)")
                    continue
                if dst.exists(): shutil.rmtree(dst)
                shutil.copytree(sub, dst)
                n = sum(1 for _ in dst.rglob("*") if _.is_file())
                print(f"    copied {sub.name}/ ({n} files)")

DATA_DIR        = WORK_DIR / "data"
CHECKPOINT_DIR  = PERSIST_DIR / "checkpoints"
RESULTS_DIR     = PERSIST_DIR / "results"
LATENT_DIR      = PERSIST_DIR / "latents"
SPLITS_DIR      = PERSIST_DIR / "splits"
EXTENDED_DIR    = PERSIST_DIR / "extended"
FIGURES_DIR     = PERSIST_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Kaggle env: {IN_KAGGLE}  PERSIST_DIR: {PERSIST_DIR}  WORK_DIR: {WORK_DIR}")

def pip_install(*pkgs):
    import subprocess
    for p in pkgs:
        try: __import__(p.split("==")[0].replace("-", "_"))
        except ImportError:
            subprocess.run(["pip", "install", "-q", p], check=True)
pip_install("pvlib", "h5py", "scikit-learn", "scipy", "tqdm",
            "opencv-python-headless", "matplotlib", "pyarrow")

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import gc, json, time, shutil, requests, math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
'''


KAGGLE_SETUP_PART2_CODE = '''\
# ==== Kaggle setup (Part 2 — also pulls in Part 1's artifacts from attached dataset) ====
import os, sys, shutil
from pathlib import Path

IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
IN_COLAB = "google.colab" in sys.modules

PERSIST_DIR = Path("/kaggle/working/solarsde_outputs") if IN_KAGGLE else (Path.cwd() / "solarsde_outputs")
WORK_DIR    = Path("/kaggle/working/solarsde") if IN_KAGGLE else (Path.cwd() / "solarsde_work")

for d in [PERSIST_DIR, WORK_DIR,
          PERSIST_DIR / "checkpoints", PERSIST_DIR / "results",
          PERSIST_DIR / "latents",     PERSIST_DIR / "splits",
          PERSIST_DIR / "extended",    PERSIST_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# Copy Part 1 outputs from any attached Kaggle dataset that looks like the
# right shape (has checkpoints/ or splits/). Skips files already present.
if IN_KAGGLE:
    src_root = Path("/kaggle/input")
    if src_root.exists():
        print("Looking for Part 1 dataset to copy ...")
        for ds in src_root.iterdir():
            if not ds.is_dir(): continue
            looks_like_part1 = (ds / "checkpoints").exists() or (ds / "splits").exists() or (ds / "latents").exists()
            if not looks_like_part1: continue
            print(f"  found candidate: {ds.name}")
            for sub in ds.iterdir():
                if not sub.is_dir(): continue
                dst = PERSIST_DIR / sub.name
                if dst.exists():
                    print(f"    skip {sub.name}/ (already in PERSIST_DIR)")
                    continue
                shutil.copytree(sub, dst)
                n = sum(1 for _ in dst.rglob("*") if _.is_file())
                print(f"    copied {sub.name}/ ({n} files)")

DATA_DIR        = WORK_DIR / "data"
CHECKPOINT_DIR  = PERSIST_DIR / "checkpoints"
RESULTS_DIR     = PERSIST_DIR / "results"
LATENT_DIR      = PERSIST_DIR / "latents"
SPLITS_DIR      = PERSIST_DIR / "splits"
EXTENDED_DIR    = PERSIST_DIR / "extended"
FIGURES_DIR     = PERSIST_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"\\nKaggle env: {IN_KAGGLE}  PERSIST_DIR: {PERSIST_DIR}")

def pip_install(*pkgs):
    import subprocess
    for p in pkgs:
        try: __import__(p.split("==")[0].replace("-", "_"))
        except ImportError:
            subprocess.run(["pip", "install", "-q", p], check=True)
pip_install("pvlib", "h5py", "scikit-learn", "scipy", "tqdm",
            "opencv-python-headless", "matplotlib", "pyarrow")

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import gc, json, time, requests, math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
'''


FAST_START_KAGGLE_CODE = '''\
# ==== Soft fast-start: try to pull cached artifacts from GitHub (best-effort) ====
import requests
GITHUB_RAW = "https://raw.githubusercontent.com/keshavkrishnan08/SDE/main"

def gh_pull_soft(rel_path, dest):
    if dest.exists() and dest.stat().st_size > 100:
        return True
    try:
        r = requests.get(f"{GITHUB_RAW}/{rel_path}", timeout=180)
        if r.status_code == 200 and len(r.content) > 100:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False

required = {
    CHECKPOINT_DIR / "vae_best.pt":   "colab_outputs/checkpoints/vae_best.pt",
    SPLITS_DIR    / "train.parquet":  "colab_outputs/splits/train.parquet",
    SPLITS_DIR    / "val.parquet":    "colab_outputs/splits/val.parquet",
    SPLITS_DIR    / "test.parquet":   "colab_outputs/splits/test.parquet",
    EXTENDED_DIR  / "train.parquet":  "colab_outputs/extended/train.parquet",
    EXTENDED_DIR  / "val.parquet":    "colab_outputs/extended/val.parquet",
    EXTENDED_DIR  / "test.parquet":   "colab_outputs/extended/test.parquet",
}
for split in ["train", "val", "test"]:
    for key in ["latents", "cti", "ghi", "covariates", "is_ramp", "kt",
                "ghi_clearsky", "physics_features"]:
        required[LATENT_DIR / f"{split}_{key}.npy"] = f"colab_outputs/latents/{split}_{key}.npy"
optional = {
    CHECKPOINT_DIR / "sde_best.pt":   "colab_outputs/checkpoints/sde_best.pt",
    CHECKPOINT_DIR / "score_best.pt": "colab_outputs/checkpoints/score_best.pt",
}
n_have = 0
for dest, rel in {**required, **optional}.items():
    if gh_pull_soft(rel, dest): n_have += 1
print(f"Fast-start: {n_have} artifacts available locally or pulled from GitHub")

HAVE_VAE     = (CHECKPOINT_DIR / "vae_best.pt").exists()
HAVE_SPLITS  = (SPLITS_DIR / "train.parquet").exists()
HAVE_LATENTS = all((LATENT_DIR / f"{s}_latents.npy").exists() for s in ["train", "val", "test"])
HAVE_KT      = all((LATENT_DIR / f"{s}_kt.npy").exists() for s in ["train", "val", "test"])
HAVE_PHYS    = all((LATENT_DIR / f"{s}_physics_features.npy").exists() for s in ["train", "val", "test"])
HAVE_EXTENDED = (EXTENDED_DIR / "train.parquet").exists()
HAVE_EXT      = HAVE_EXTENDED   # alias for backward compat with LOAD_DATA_TOLERANT_CODE
NEED_GOLDEN_RETRAIN = not (HAVE_VAE and HAVE_SPLITS and HAVE_LATENTS and HAVE_KT and HAVE_PHYS)
print(f"NEED_GOLDEN_RETRAIN = {NEED_GOLDEN_RETRAIN}")

# ==== Image-features gate (independent of Golden retrain) ====
# True if we don't yet have real (non-zero) image features. The CloudCV
# download stage will run if this is True, even when NEED_GOLDEN_RETRAIN is
# False. Lets the user resume from cached latents but still extract real
# image features (optical flow + sun-ROI + cloud fraction) instead of falling
# back to zero-fill.
def _have_real_image_features():
    import numpy as _np
    for _s in ["train", "val", "test"]:
        _p = LATENT_DIR / f"{_s}_image_features.npy"
        if not _p.exists():
            return False
        _a = _np.load(_p, mmap_mode="r")
        if _a.size == 0 or not _a.any():   # all-zero zero-fill counts as "not real"
            return False
    return True
NEED_IMAGE_FEATURES = not _have_real_image_features()
print(f"NEED_IMAGE_FEATURES = {NEED_IMAGE_FEATURES}  "
      f"(triggers CloudCV download even if Golden retrain is skipped)")

# ==== Stage-0 retraining gate (consumed by STAGE0_CODE) ====
SDE_CKPT   = CHECKPOINT_DIR / "sde_best.pt"
SCORE_CKPT = CHECKPOINT_DIR / "score_best.pt"
NEED_NB2_TRAINING = not (SDE_CKPT.exists() and SCORE_CKPT.exists())
print(f"NEED_NB2_TRAINING   = {NEED_NB2_TRAINING}  (SDE+Score will be trained inline if True)")
'''


PREREQ_CHECK_PART2_CODE = '''\
# ==== Prerequisite check ====
_missing = []
for p in [CHECKPOINT_DIR / "vae_best.pt",
          CHECKPOINT_DIR / "sde_best.pt",
          CHECKPOINT_DIR / "score_best.pt",
          LATENT_DIR / "test_latents.npy",
          LATENT_DIR / "test_kt.npy",
          LATENT_DIR / "test_physics_features.npy",
          SPLITS_DIR / "test.parquet"]:
    if not p.exists():
        _missing.append(str(p))
if _missing:
    msg = ("This notebook expects Part 1 (08a_master_part1_kaggle.ipynb) "
           "artifacts to be available.\\nMissing:\\n  " + "\\n  ".join(_missing) +
           "\\n\\nFix: Add Data → Your Datasets → attach your 08a saved version.")
    raise RuntimeError(msg)
print("All Part 1 artifacts present. Ready to proceed.")
'''


def nb_08a_kaggle():
    """Part 1: data + VAE + training + 5-fold CV."""
    cells = [
        ("markdown", HEADER_08A_MD),
        ("markdown", "## 0. Setup (Kaggle)"),
        ("code", KAGGLE_SETUP_PART1_CODE),
        ("code", FAST_START_KAGGLE_CODE),
        ("markdown", "## 0a. Preflight sanity check (catches missing names + corrupt ckpts early)"),
        ("code", PREFLIGHT_SANITY_CODE),

        ("markdown", "## 1. Shared model definitions"),
        ("code", SHARED_CODE),

        ("markdown", "## RETRAIN — Golden CO (skipped if cached)"),
        ("code", GOLDEN_RETRAIN_GUARD_CODE),
        ("code", "LATENT_DIM = 64\nIMG_SIZE = 128\n" + VAE_MODEL),
        # CloudCV download is now gated on BOTH ENABLE_GOLDEN_RETRAIN OR
        # NEED_IMAGE_FEATURES — so even when the user has cached latents +
        # ckpts (Golden retrain off), the raw images get downloaded if image
        # features are still zero-fill / missing.
        ("code", _gate("(ENABLE_GOLDEN_RETRAIN or NEED_IMAGE_FEATURES) and not all((DATA_DIR / 'cloudcv' / f).exists() "
                       "for f in ['2019_09_07.tar.gz'])", CLOUDCV_DOWNLOAD_ROBUST)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN or NEED_IMAGE_FEATURES", CLOUDCV_EXTRACT_ROBUST)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (DATA_DIR / 'bms' / 'bms_srrl_2019.csv').exists()",
                       BMS_DOWNLOAD)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (SPLITS_DIR / 'train.parquet').exists()",
                       PREPROCESS_CODE)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN", IMAGE_DATASET)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (CHECKPOINT_DIR / 'vae_best.pt').exists()",
                       VAE_TRAIN)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (LATENT_DIR / 'test_latents.npy').exists()",
                       LATENT_EXTRACT)),
        ("code", GOLDEN_KT_PHYS_CODE),
        ("code", GOLDEN_EXTENDED_CODE),

        ("markdown", "## STAGE B — Image-feature pre-flight + extraction"),
        # Fallback runs BEFORE LOAD_DATA so cov dim is consistent from the
        # very first load — and any stale zero-fill that would mismatch an
        # existing SDE ckpt's c_dim is deleted up front.
        ("code", STAGE_M1_SAFE_FALLBACK_CODE),
        ("code", STAGE_MINUS1_CODE),

        ("markdown", "## 2. Load data tensors"),
        ("code", LOAD_DATA_TOLERANT_CODE),

        ("markdown", "## STAGE C — Train Latent Neural SDE (Mixture-of-OU, closed-form marginals, CTI-gated diffusion)"),
        # SolarSDE v2: replaces the SDE+score-decoder with a CTI-gated mixture
        # density network anchored on smart-persistence. Guaranteed >= persistence
        # by construction, calibrated PICP, trains in ~10 min on a T4.
        ("code", MDN_ARCHITECTURE_CODE),
        ("code", STAGE_0_V2_CODE),
        ("code", safe_stage("POST_STAGE0_V2_VERIFY", POST_STAGE0_V2_VERIFY_CODE)),

        ("markdown", "## STAGE CV — Leave-one-day-out cross-validation (reviewer requirement)"),
        ("code", safe_stage("K_FOLD_CV", K_FOLD_CV_CODE)),

        ("markdown", "## Final — Zip Part 1 outputs to /kaggle/working/"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


def nb_08b_kaggle():
    """Part 2: baselines + ablations + stats + figures + tables."""
    cells = [
        ("markdown", HEADER_08B_MD),
        ("markdown", "## 0. Setup (Kaggle — Part 2)"),
        ("code", KAGGLE_SETUP_PART2_CODE),
        ("code", FAST_START_KAGGLE_CODE),
        ("markdown", "## 0a. Preflight sanity check"),
        ("code", PREFLIGHT_SANITY_CODE),
        ("markdown", "## Prerequisite check"),
        ("code", PREREQ_CHECK_PART2_CODE),

        ("markdown", "## 1. Shared model definitions"),
        ("code", SHARED_CODE),

        ("markdown", "## 2. Load data tensors"),
        ("code", LOAD_DATA_TOLERANT_CODE),

        ("markdown", "## STAGE C+ — Corrected inference (advance time-deterministic covariates)"),
        ("code", safe_stage("CORRECTED_INFERENCE", CORRECTED_INFERENCE_CODE)),

        ("markdown", "## STAGE D — Standard baselines (persistence, smart-pers, LSTM, MC-Dropout, CSDI)"),
        ("code", safe_stage("BASELINES", BASELINES_CODE)),

        ("markdown", "## STAGE F — Ablations A2 (no-CTI), A4 (no-score), A5 (no-SDE/ODE), A3 (no-VAE PCA)"),
        ("code", safe_stage("ABLATIONS", ABLATIONS_CODE)),
        ("code", safe_stage("EXTRA_ABLATIONS", EXTRA_ABLATIONS_CODE)),

        ("markdown", "## STAGE G — Conformal calibration"),
        ("code", safe_stage("CALIBRATION", CALIBRATION_CODE)),

        ("markdown", "## STAGE H — Stratified eval + Diebold-Mariano test"),
        ("code", safe_stage("STRATIFIED", STRATIFIED_CODE)),

        ("markdown", "## STAGE I — PIT + reliability + sharpness + bootstrap CIs (all horizons)"),
        ("code", safe_stage("PIT_RELIABILITY", PIT_RELIABILITY_CODE)),
        ("code", safe_stage("BOOTSTRAP_CIS", BOOTSTRAP_CIS_CODE)),

        ("markdown", "## STAGE I+ — Ramp detection AUROC + CTI lead-time"),
        ("code", safe_stage("RAMP_AUROC", RAMP_AUROC_CODE)),

        ("markdown", "## STAGE I++ — CTI vs cloud-cover validation (physical-meaningfulness)"),
        ("code", safe_stage("CTI_VALIDATION", CTI_VALIDATION_CODE)),

        ("markdown", "## STAGE I+++ — Holm-Bonferroni multiple-comparison correction"),
        ("code", safe_stage("HOLM_BONFERRONI", HOLM_BONFERRONI_CODE)),

        ("markdown", "## STAGE K — Resource-cost simulation (CAISO reserves, USD/yr per GW)"),
        ("code", safe_stage("ECONOMIC_CAISO", ECONOMIC_CAISO_CODE)),

        ("markdown", "## STAGE M — Analysis figures (CTI dynamics, regime, forecast traces)"),
        ("code", safe_stage("ANALYSIS", ANALYSIS_CODE)),

        ("markdown", "## STAGE N — LaTeX tables (3 paper-ready tables for Solar Energy)"),
        ("code", safe_stage("LATEX_TABLES", LATEX_TABLES_CODE)),

        ("markdown", "## Final — Zip the paper package to /kaggle/working/"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


HEADER_08_COMBINED_MD = """# SolarSDE Master (Combined) — single Kaggle notebook end-to-end

This is the merger of `08a_master_part1_kaggle.ipynb` (data + VAE + STAGE 0 +
5-fold CV) and `08b_master_part2_kaggle.ipynb` (baselines + ablations +
calibration + figures + tables) into a single notebook so you can run the
whole paper pipeline in one Kaggle session.

**Heads-up on Kaggle session limits.**  Free-tier Kaggle GPU sessions are
capped at 12 hours; T4-x2 at 9 hours.  Running everything end-to-end on a
fresh notebook takes 9-14 h depending on the GPU, so you may still hit the
wall.  Mitigations:

1. Every stage auto-resumes from disk — if the kernel dies, re-run all cells
   and finished stages skip.  PERSIST_DIR lives at /kaggle/working/solarsde_outputs
   and is preserved between sessions when you Save Version → Save & Run All.
2. To split into 2 sessions, run cells through STAGE CV first, Save Version,
   then re-attach the saved dataset and re-run; everything past STAGE 0 will
   then start fresh while the heavy training is already done.
3. If you want the original 2-notebook layout back, use 08a + 08b instead.

## Kaggle setup
1. Settings → Accelerator: **GPU P100** (preferred) or **T4 x2**
2. Settings → Internet: **On**
3. Run all cells
4. When done: download `/kaggle/working/solarsde_outputs.zip` from the Output tab
"""


def nb_08_kaggle_combined():
    """Single combined notebook = 08a stages + 08b stages, one Kaggle session."""
    cells = [
        ("markdown", HEADER_08_COMBINED_MD),
        ("markdown", "## 0. Setup (Kaggle)"),
        ("code", KAGGLE_SETUP_PART1_CODE),
        ("code", FAST_START_KAGGLE_CODE),
        ("markdown", "## 0a. Preflight sanity check"),
        ("code", PREFLIGHT_SANITY_CODE),

        ("markdown", "## 1. Shared model definitions"),
        ("code", SHARED_CODE),

        ("markdown", "## RETRAIN — Golden CO (skipped if cached)"),
        ("code", GOLDEN_RETRAIN_GUARD_CODE),
        ("code", "LATENT_DIM = 64\nIMG_SIZE = 128\n" + VAE_MODEL),
        ("code", _gate("(ENABLE_GOLDEN_RETRAIN or NEED_IMAGE_FEATURES) and not all((DATA_DIR / 'cloudcv' / f).exists() "
                       "for f in ['2019_09_07.tar.gz'])", CLOUDCV_DOWNLOAD_ROBUST)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN or NEED_IMAGE_FEATURES", CLOUDCV_EXTRACT_ROBUST)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (DATA_DIR / 'bms' / 'bms_srrl_2019.csv').exists()",
                       BMS_DOWNLOAD)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (SPLITS_DIR / 'train.parquet').exists()",
                       PREPROCESS_CODE)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN", IMAGE_DATASET)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (CHECKPOINT_DIR / 'vae_best.pt').exists()",
                       VAE_TRAIN)),
        ("code", _gate("ENABLE_GOLDEN_RETRAIN and not (LATENT_DIR / 'test_latents.npy').exists()",
                       LATENT_EXTRACT)),
        ("code", GOLDEN_KT_PHYS_CODE),
        ("code", GOLDEN_EXTENDED_CODE),

        ("markdown", "## STAGE B — Image-feature pre-flight + extraction"),
        ("code", STAGE_M1_SAFE_FALLBACK_CODE),
        ("code", STAGE_MINUS1_CODE),

        ("markdown", "## 2. Load data tensors"),
        ("code", LOAD_DATA_TOLERANT_CODE),

        ("markdown", "## STAGE C — Train Latent Neural SDE (Mixture-of-OU, closed-form marginals, CTI-gated diffusion)"),
        ("code", MDN_ARCHITECTURE_CODE),
        ("code", STAGE_0_V2_CODE),
        ("code", safe_stage("POST_STAGE0_V2_VERIFY", POST_STAGE0_V2_VERIFY_CODE)),

        ("markdown", "## STAGE CV — Leave-one-day-out cross-validation"),
        ("code", safe_stage("K_FOLD_CV", K_FOLD_CV_CODE)),

        ("markdown", "## STAGE C+ — Corrected inference (advance time-deterministic covariates)"),
        ("code", safe_stage("CORRECTED_INFERENCE", CORRECTED_INFERENCE_CODE)),

        ("markdown", "## STAGE D — Standard baselines (persistence, smart-pers, LSTM, MC-Dropout, CSDI)"),
        ("code", safe_stage("BASELINES", BASELINES_CODE)),

        ("markdown", "## STAGE F — Ablations A2 (no-CTI), A3 (no-VAE PCA), A4 (no-score), A5 (no-SDE/ODE)"),
        ("code", safe_stage("ABLATIONS", ABLATIONS_CODE)),
        ("code", safe_stage("EXTRA_ABLATIONS", EXTRA_ABLATIONS_CODE)),

        ("markdown", "## STAGE G — Conformal calibration"),
        ("code", safe_stage("CALIBRATION", CALIBRATION_CODE)),

        ("markdown", "## STAGE H — Stratified eval + Diebold-Mariano test"),
        ("code", safe_stage("STRATIFIED", STRATIFIED_CODE)),

        ("markdown", "## STAGE I — PIT + reliability + sharpness + bootstrap CIs"),
        ("code", safe_stage("PIT_RELIABILITY", PIT_RELIABILITY_CODE)),
        ("code", safe_stage("BOOTSTRAP_CIS", BOOTSTRAP_CIS_CODE)),

        ("markdown", "## STAGE I+ — Ramp detection AUROC + CTI lead-time"),
        ("code", safe_stage("RAMP_AUROC", RAMP_AUROC_CODE)),

        ("markdown", "## STAGE I++ — CTI vs cloud-cover validation"),
        ("code", safe_stage("CTI_VALIDATION", CTI_VALIDATION_CODE)),

        ("markdown", "## STAGE I+++ — Holm-Bonferroni multiple-comparison correction"),
        ("code", safe_stage("HOLM_BONFERRONI", HOLM_BONFERRONI_CODE)),

        ("markdown", "## STAGE K — CAISO economic value (USD/yr per GW)"),
        ("code", safe_stage("ECONOMIC_CAISO", ECONOMIC_CAISO_CODE)),

        ("markdown", "## STAGE M — Analysis figures (CTI dynamics, regime, forecast traces)"),
        ("code", safe_stage("ANALYSIS", ANALYSIS_CODE)),

        ("markdown", "## STAGE N — LaTeX tables (3 paper-ready tables for Solar Energy)"),
        ("code", safe_stage("LATEX_TABLES", LATEX_TABLES_CODE)),

        ("markdown", "## Final — Zip the paper package to /kaggle/working/"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    for name, builder in [
        ("08a_master_part1_kaggle.ipynb", nb_08a_kaggle),
        ("08b_master_part2_kaggle.ipynb", nb_08b_kaggle),
        ("08_master_kaggle.ipynb",        nb_08_kaggle_combined),
    ]:
        path = NB_DIR / name
        nb = builder()
        path.write_text(json.dumps(nb, indent=1))
        print(f"Wrote {name}: {path.stat().st_size / 1024:.1f} KB ({len(nb['cells'])} cells)")
