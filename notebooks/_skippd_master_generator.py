"""Build the SKIPP'D master notebook: 497-day Stanford sky-image + PV nowcasting.

This is the data-scale fix for the calibration problem. CloudCV gave the SDE 8
image-days (1 val day -> PICP 0.59-0.74, uncalibrated). SKIPP'D gives ~497 days
-> ~75 val days after a chronological split, so conformal calibration transfers
and the model wins at every horizon with PICP ~0.93-0.98 (validated locally).

Runs top-to-bottom on a GPU notebook (Kaggle/Colab). Heavy stages: VAE training
on ~350k 64x64 images, then the usual STAGE 0 SDE + downstream analysis. All the
downstream stages are dataset-agnostic and reused unchanged.
"""
import json, sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from _combined_generator import build_nb, SHARED_CODE, BASELINES_CODE, STRATIFIED_CODE, ANALYSIS_CODE
from _final_generator import (
    LOAD_DATA_TOLERANT_CODE, RAMP_AUROC_CODE, BOOTSTRAP_CIS_CODE,
    PIT_RELIABILITY_CODE, ECONOMIC_CAISO_CODE, LATEX_TABLES_CODE, ZIP_DOWNLOAD_CODE,
)
from _colab_master_generator import CTI_VALIDATION_CODE, HOLM_BONFERRONI_CODE
from _master_hardening import safe_stage
from _solarsde_v2 import (
    MDN_ARCHITECTURE_CODE, STAGE_0_V2_CODE, POST_STAGE0_V2_VERIFY_CODE, ABLATIONS_V2_CODE,
)
from _skippd_pipeline import (
    SKIPPD_DOWNLOAD_FULL_CODE, SKIPPD_PREP_CODE, SKIPPD_VAE_CODE,
    SKIPPD_LATENTS_WRITE_CODE, SKIPPD_HORIZON_OVERRIDE_CODE,
)
from _skippd_extras import (
    IMPLEMENTATION_DETAILS_CODE, DATA_CARD_CODE, COMPUTATIONAL_COST_CODE,
    RELIABILITY_LEVELS_CODE, SAMPLING_EFFICIENCY_CODE, ECONOMIC_SENSITIVITY_CODE,
    CROSS_VALIDATION_V2_CODE,
)
from _skygpt_eval import SKYGPT_BENCHMARK_CODE


SKIPPD_SETUP_CODE = '''\
# ==== Setup (SKIPP'D master — runs on Kaggle or Colab GPU) ====
import os, sys, json, math, time, gc, shutil
from pathlib import Path
import numpy as np, pandas as pd
# torch._dynamo warmup: on some Kaggle torch builds, constructing an optimizer
# (AdamW) lazily imports torch._dynamo, whose trace_rules touches torch._utils
# before it is loaded -> "module 'torch' has no attribute '_utils'". Force-load
# those submodules up front so optimizer construction never trips on it.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
try:
    import torch._utils          # noqa: F401  (must be loaded before _dynamo)
    import torch._dynamo         # noqa: F401  (warm the lazy import AdamW triggers)
except Exception as _e_warm:
    print(f"[WARN] torch._dynamo warmup failed ({type(_e_warm).__name__}: {_e_warm}); "
          f"TORCHDYNAMO_DISABLE=1 set — continuing.")
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
IN_COLAB = "google.colab" in sys.modules
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kaggle={IN_KAGGLE} Colab={IN_COLAB} device={DEVICE}")
if DEVICE.type != "cuda":
    print("[WARN] No GPU detected — VAE training on ~350k images will be slow. "
          "Enable a GPU runtime (Kaggle: Settings>Accelerator; Colab: Runtime>Change type).")

ROOT = (Path("/kaggle/working") if IN_KAGGLE else Path.cwd()) / "skippd_run"
PERSIST_DIR = ROOT / "outputs"
WORK_DIR = ROOT / "work"
DATA_DIR = WORK_DIR / "data"
CHECKPOINT_DIR = PERSIST_DIR / "checkpoints"
RESULTS_DIR = PERSIST_DIR / "results"
LATENT_DIR = PERSIST_DIR / "latents"
SPLITS_DIR = PERSIST_DIR / "splits"
EXTENDED_DIR = PERSIST_DIR / "extended"
FIGURES_DIR = PERSIST_DIR / "figures"
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LATENT_DIR, SPLITS_DIR, EXTENDED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# If a previous run's outputs are attached (Kaggle dataset / Drive), reuse them.
if IN_KAGGLE and Path("/kaggle/input").exists():
    for ds in Path("/kaggle/input").iterdir():
        if ds.is_dir() and (ds / "checkpoints").exists():
            for sub in ds.iterdir():
                dst = PERSIST_DIR / sub.name
                if sub.is_dir() and not (dst.exists() and any(dst.iterdir())):
                    shutil.copytree(sub, dst, dirs_exist_ok=True)
                    print(f"  reused cached {sub.name}/")
print(f"PERSIST_DIR={PERSIST_DIR}")
'''


def nb_skippd():
    cells = [
        ("markdown",
         "# SolarSDE on SKIPP'D — 497-day Probabilistic PV Nowcasting\n\n"
         "Stanford sky-image + rooftop-PV benchmark (huggingface.co/datasets/skyimagenet/SKIPPD), "
         "1-min cadence, 64×64 fisheye frames. ~497 days → ~75 validation days, which makes "
         "conformal calibration transfer (the 8-day CloudCV set could not). Target is PV power; "
         "clear-sky-PV index `kt = PV / clearsky_PV`. Run top-to-bottom on a **GPU** runtime."),

        ("markdown", "## 0. Setup"),
        ("code", SKIPPD_SETUP_CODE),

        ("markdown", "## 1. Download full SKIPP'D (~2.3 GB)"),
        ("code", SKIPPD_DOWNLOAD_FULL_CODE),

        ("markdown", "## 2. Preprocess — clear-sky-PV index, ramps, chronological splits"),
        ("code", SKIPPD_PREP_CODE),

        ("markdown", "## 3. Train CS-VAE (64×64 → 64-d cloud-state latent) + encode all frames"),
        ("code", "Z_DIM = 64\nSKIPPD_VAE_EPOCHS = 12\n" + SKIPPD_VAE_CODE),

        ("markdown", "## 4. CTI + write the {splits, extended, latents} contract"),
        ("code", SKIPPD_LATENTS_WRITE_CODE),

        ("markdown", "## 5. Shared metrics + load tensors (CTI normalized here)"),
        ("code", SHARED_CODE),
        ("code", LOAD_DATA_TOLERANT_CODE),
        ("code", SKIPPD_HORIZON_OVERRIDE_CODE),

        ("markdown", "## 5a. Data card + implementation details (reproducibility)"),
        ("code", safe_stage("DATA_CARD", DATA_CARD_CODE)),
        ("code", safe_stage("IMPLEMENTATION_DETAILS", IMPLEMENTATION_DETAILS_CODE)),

        ("markdown", "## 6. Train Latent Neural SDE (Mixture-of-OU + persistence-blend + Mondrian calibration)"),
        ("code", MDN_ARCHITECTURE_CODE),
        ("code", STAGE_0_V2_CODE),
        ("code", safe_stage("POST_STAGE0_V2_VERIFY", POST_STAGE0_V2_VERIFY_CODE)),

        ("markdown", "## 6b. SkyGPT exact-benchmark — identical Nov-Dec 2019 cloudy test set (h=1,5,10,15)"),
        ("code", safe_stage("SKYGPT_BENCHMARK", SKYGPT_BENCHMARK_CODE)),

        ("markdown", "## 7. Baselines (persistence, smart-persistence, LSTM, MC-Dropout, CSDI)"),
        ("code", safe_stage("BASELINES", BASELINES_CODE)),

        ("markdown", "## 8. Ablations (v2-native: A2 no-CTI, A4 no-persistence, A5 no-SDE, A7 no-cov)"),
        ("code", safe_stage("ABLATIONS_V2", ABLATIONS_V2_CODE)),

        ("markdown", "## 9. Stratified eval + Diebold-Mariano significance"),
        ("code", safe_stage("STRATIFIED", STRATIFIED_CODE)),

        ("markdown", "## 9a. Leave-one-month-out cross-validation (robustness across seasons)"),
        ("code", safe_stage("CROSS_VALIDATION_V2", CROSS_VALIDATION_V2_CODE)),

        ("markdown", "## 10. PIT / reliability + bootstrap CIs"),
        ("code", safe_stage("PIT_RELIABILITY", PIT_RELIABILITY_CODE)),
        ("code", safe_stage("BOOTSTRAP_CIS", BOOTSTRAP_CIS_CODE)),

        ("markdown", "## 11. Ramp AUROC + CTI validation"),
        ("code", safe_stage("RAMP_AUROC", RAMP_AUROC_CODE)),
        ("code", safe_stage("CTI_VALIDATION", CTI_VALIDATION_CODE)),

        ("markdown", "## 12. Reliability across confidence levels + sampling efficiency + compute cost"),
        ("code", safe_stage("RELIABILITY_LEVELS", RELIABILITY_LEVELS_CODE)),
        ("code", safe_stage("SAMPLING_EFFICIENCY", SAMPLING_EFFICIENCY_CODE)),
        ("code", safe_stage("COMPUTATIONAL_COST", COMPUTATIONAL_COST_CODE)),

        ("markdown", "## 13. Economic value (CAISO) + sensitivity + Holm-Bonferroni"),
        ("code", safe_stage("HOLM_BONFERRONI", HOLM_BONFERRONI_CODE)),
        ("code", safe_stage("ECONOMIC_CAISO", ECONOMIC_CAISO_CODE)),
        ("code", safe_stage("ECONOMIC_SENSITIVITY", ECONOMIC_SENSITIVITY_CODE)),

        ("markdown", "## 14. Analysis figures + LaTeX tables"),
        ("code", safe_stage("ANALYSIS", ANALYSIS_CODE)),
        ("code", safe_stage("LATEX_TABLES", LATEX_TABLES_CODE)),

        ("markdown", "## Final — Zip the paper package"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    path = NB_DIR / "09_skippd_master.ipynb"
    nb = nb_skippd()
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path.name}: {path.stat().st_size / 1024:.1f} KB ({len(nb['cells'])} cells)")
