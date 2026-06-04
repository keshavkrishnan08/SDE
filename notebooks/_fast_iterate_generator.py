"""Build notebook 12: FAST ITERATION notebook.

The lean loop for iterating on the model: data -> closed-form train ->
all-horizon results + SkyGPT head-to-head. NOTHING else (no rollout, no
baselines, ablations, CV, economics). ~1.5 h/run instead of ~5 h, so you can
quickly test architecture/calibration changes against the numbers that matter:
  - all-weather per-horizon CRPS / PICP (printed by STAGE 0)
  - SkyGPT exact cloudy test per-horizon CRPS / Winkler + h=15 head-to-head

Same GitHub-pull provenance and failure-isolation as notebook 11. Set
TRAIN_ROLLOUT=True in the config cell if you also want the rollout variant.
"""
import json, sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))
from _combined_generator import build_nb
from _final_master_generator import CLONE_AND_IMPORT_CODE, _cell, _cell_raw

REPO_HTTPS = "https://github.com/keshavkrishnan08/SDE.git"
REPO_ZIP   = "https://github.com/keshavkrishnan08/SDE/archive/refs/heads/main.zip"


FAST_SETUP_CODE = '''\
# ==== Setup (fast-iteration: closed-form only, all-horizon + SkyGPT) ====
import os, sys, json, math, time, gc, shutil, subprocess, traceback
from pathlib import Path
import numpy as np, pandas as pd
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import torch
try:
    import torch._utils, torch._dynamo  # noqa: F401
except Exception as _e:
    print(f"[WARN] dynamo warmup: {_e} — continuing")
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
IN_COLAB = "google.colab" in sys.modules
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kaggle={IN_KAGGLE} Colab={IN_COLAB} device={DEVICE}")
if DEVICE.type != "cuda":
    print("[WARN] No GPU — enable a GPU runtime.")

ROOT = (Path("/kaggle/working") if IN_KAGGLE else Path.cwd()) / "fast_run"
PERSIST_DIR = ROOT / "outputs"; WORK_DIR = ROOT / "work"; DATA_DIR = WORK_DIR / "data"
CHECKPOINT_DIR = PERSIST_DIR / "checkpoints"; RESULTS_DIR = PERSIST_DIR / "results"
LATENT_DIR = PERSIST_DIR / "latents"; SPLITS_DIR = PERSIST_DIR / "splits"
EXTENDED_DIR = PERSIST_DIR / "extended"; FIGURES_DIR = PERSIST_DIR / "figures"
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LATENT_DIR, SPLITS_DIR, EXTENDED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== Config — iterate on these =====
Z_DIM = 64
SKIPPD_VAE_EPOCHS = 12     # CS-VAE (cached after first run if you attach outputs)
CLOSEDFORM_EPOCHS = 60     # the knob you'll mostly tune
TRAIN_ROLLOUT     = False  # set True to also train + ensemble the rollout variant
ARCH = "base"              # which architecture to train + benchmark on SkyGPT.
                           # Options: base | bigmix | wide | deep | gru
                           #   base   K=3 d=128 L=2  (reference closed-form)
                           #   bigmix K=8 d=128 L=3  (heavy-tail mixture for cloudy ramps)
                           #   wide   K=4 d=256 L=2  (wider transformer)
                           #   deep   K=4 d=128 L=4  (deeper transformer)
                           #   gru    GRU encoder    (different temporal bias)
MOTION_GRID = 1            # SPATIAL cloud-motion features (the real beat-attempt):
                           #   1 = global-mean optical flow (4 dims, current)
                           #   3 = 3x3 grid-pooled flow (27 dims) — keeps WHERE clouds
                           #       move toward the sun, the spatial signal SkyGPT uses
                           #   4 = 4x4 grid (48 dims)
                           # Sweep ARCH x MOTION_GRID; if a combo beats 2.81 at h=15,
                           # set the same knobs in notebook 11 for the final paper run.
SEED_ENSEMBLE     = 1      # >1 = deep ensemble: train this many closed-form models
                           # with different seeds and pool their samples. The one
                           # legitimate lever with a real shot at beating SkyGPT
                           # (deep ensembles reliably cut CRPS ~5-12%). Costs
                           # SEED_ENSEMBLE x ~85 min — set 3 when you want the real try.

# Reuse a cached VAE/latents from an attached Kaggle dataset to skip ~25 min.
if IN_KAGGLE and Path("/kaggle/input").exists():
    for ds in Path("/kaggle/input").iterdir():
        if ds.is_dir() and (ds / "checkpoints" / "skippd_vae.pt").exists():
            for sub in ["checkpoints", "latents", "splits", "extended"]:
                s = ds / sub
                if s.exists(): shutil.copytree(s, PERSIST_DIR / sub, dirs_exist_ok=True)
            print(f"  reused cached VAE/latents from {ds.name} — VAE+motion will skip")
print(f"Config: VAE={SKIPPD_VAE_EPOCHS}ep, closed-form={CLOSEDFORM_EPOCHS}ep, rollout={TRAIN_ROLLOUT}")
print(f"PERSIST_DIR={PERSIST_DIR}")
'''

# Trim the clone cell's import list to only what the fast loop needs (+ SkyGPT eval).
FAST_CLONE_CODE = CLONE_AND_IMPORT_CODE.replace(
    'globals().update(_safe_import("_solarsde_rollout",\n'
    '    ["ROLLOUT_ARCH_CODE", "ABLATIONS_ROLLOUT_CODE"]))',
    'globals().update(_safe_import("_solarsde_rollout",\n'
    '    ["ROLLOUT_ARCH_CODE"]))\n'
    'globals().update(_safe_import("_skygpt_eval", ["SKYGPT_BENCHMARK_CODE"]))\n'
    'globals().update(_safe_import("_skygpt_sweep", ["SKYGPT_SWEEP_CODE", "DEEP_ENSEMBLE_CODE"]))\n'
    'globals().update(_safe_import("_arch_variants", ["ARCH_VARIANTS_CODE"]))\n'
    'globals().update(_safe_import("_ensemble_eval",\n'
    '    ["STASH_CLOSEDFORM_CODE", "STASH_ROLLOUT_CODE", "CHAMPION_SELECT_CODE",\n'
    '     "SKYGPT_TRIPLE_BENCHMARK_CODE"]))')


def nb_fast():
    cells = [
        ("markdown",
         "# SolarSDE — Fast Iteration Notebook\n\n"
         "Lean loop for tuning the model: **data → closed-form train → all-horizon results "
         "+ SkyGPT head-to-head.** No rollout (unless `TRAIN_ROLLOUT=True`), no baselines/ablations/"
         "CV/economics — those live in `11_final_publication.ipynb`.\n\n"
         "**Produces:** all-weather per-horizon CRPS/PICP (STAGE 0) and the SkyGPT exact-cloudy-test "
         "per-horizon CRPS/Winkler + h=15 head-to-head vs the published 2.81. ~1.5 h/run on a T4.\n\n"
         "Code pulled live from github.com/keshavkrishnan08/SDE."),

        ("markdown", "## 0. Environment + config"),
        ("code", FAST_SETUP_CODE),
        ("markdown", "## 1. Pull code from GitHub"),
        ("code", FAST_CLONE_CODE),

        ("markdown", "## 2. Data: SKIPP'D (~2.3 GB) + SkyGPT test set"),
        ("code", _cell("DOWNLOAD_SKIPPD", "SKIPPD_DOWNLOAD_FULL_CODE")),
        ("markdown", "## 3. Preprocess"),
        ("code", _cell("SKIPPD_PREP", "SKIPPD_PREP_CODE")),
        ("markdown", "## 4. CS-VAE + encode + optical-flow motion (skips if cached)"),
        ("code", _cell("SKIPPD_VAE", "SKIPPD_VAE_CODE")),
        ("markdown", "## 5. CTI + write contract"),
        ("code", _cell("SKIPPD_WRITE", "SKIPPD_LATENTS_WRITE_CODE")),
        ("markdown", "## 6. Shared + load + 1-min horizon config"),
        ("code", _cell("SHARED", "SHARED_CODE")),
        ("code", _cell("LOAD_DATA", "LOAD_DATA_TOLERANT_CODE")),
        ("code", _cell("HORIZON_OVERRIDE", "SKIPPD_HORIZON_OVERRIDE_CODE")),

        ("markdown", "## 7. Architecture (select via ARCH) + train (all-weather per-horizon prints here)"),
        ("code", _cell("CLOSEDFORM_ARCH", "MDN_ARCHITECTURE_CODE")),
        ("code", _cell("ARCH_SELECT", "ARCH_VARIANTS_CODE")),
        ("code", _cell_raw("CLOSEDFORM_GLUE",
                           "# ClosedFormSDE is set by ARCH_SELECT; fall back to base if that stage was skipped\n"
                           "if 'ClosedFormSDE' not in globals(): ClosedFormSDE = TemporalLatentSDE\n"
                           "print('arch ready:', globals().get('ARCH','base'))")),
        ("code", _cell_raw("CLOSEDFORM_TRAIN",
                           "exec(safe_stage('STAGE0_CLOSEDFORM',\n"
                           "     STAGE_0_V2_CODE.replace('EPOCHS = 60', f'EPOCHS = {CLOSEDFORM_EPOCHS}')), globals())")),
        ("code", _cell("CLOSEDFORM_VERIFY", "POST_STAGE0_V2_VERIFY_CODE")),

        ("markdown", "## 8. (optional) Rollout variant + ensemble — only if TRAIN_ROLLOUT"),
        ("code", _cell_raw("OPTIONAL_ROLLOUT",
                           "if not globals().get('TRAIN_ROLLOUT', False):\n"
                           "    print('[SKIP] TRAIN_ROLLOUT=False — closed-form only (fast path).')\n"
                           "else:\n"
                           "    exec(safe_stage('STASH_CLOSEDFORM', STASH_CLOSEDFORM_CODE), globals())\n"
                           "    exec(safe_stage('ROLLOUT_ARCH', ROLLOUT_ARCH_CODE), globals())\n"
                           "    exec(safe_stage('STAGE0_ROLLOUT',\n"
                           "         STAGE_0_V2_CODE.replace('EPOCHS = 60', 'EPOCHS = 35')), globals())\n"
                           "    exec(safe_stage('STASH_ROLLOUT', STASH_ROLLOUT_CODE), globals())\n"
                           "    exec(safe_stage('CHAMPION_SELECT', CHAMPION_SELECT_CODE), globals())")),

        ("markdown", "## 8b. (optional) Deep ensemble — train K seeds (the real lever; SEED_ENSEMBLE>1)"),
        ("code", _cell("DEEP_ENSEMBLE", "DEEP_ENSEMBLE_CODE")),

        ("markdown", "## 9. SkyGPT exact-benchmark — all horizons + h=15 head-to-head"),
        ("code", _cell_raw("SKYGPT",
                           "# triple benchmark if rollout trained, else single-model on the champion\n"
                           "if globals().get('TRAIN_ROLLOUT', False) and (CHECKPOINT_DIR/'mdn_rollout_best.pt').exists():\n"
                           "    exec(safe_stage('SKYGPT', SKYGPT_TRIPLE_BENCHMARK_CODE), globals())\n"
                           "else:\n"
                           "    exec(safe_stage('SKYGPT', SKYGPT_BENCHMARK_CODE), globals())")),

        ("markdown", "## 9b. Idea SWEEP — post-hoc knobs (val-selected + test-oracle diagnostic)"),
        ("code", _cell("SKYGPT_SWEEP", "SKYGPT_SWEEP_CODE")),

        ("markdown", "## 10. Save results"),
        ("code", _cell_raw("SAVE",
                           "import shutil\n"
                           "out = (Path('/kaggle/working') if IN_KAGGLE else Path.cwd()) / 'fast_results.zip'\n"
                           "shutil.make_archive(str(out)[:-4], 'zip', RESULTS_DIR)\n"
                           "print(f'results zipped -> {out}')\n"
                           "for f in sorted(RESULTS_DIR.glob('*.csv')):\n"
                           "    print(' ', f.name)")),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    path = NB_DIR / "12_fast_iterate.ipynb"
    nb = nb_fast()
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path.name}: {path.stat().st_size / 1024:.1f} KB ({len(nb['cells'])} cells)")
