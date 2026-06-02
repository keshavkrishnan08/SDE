"""Build notebook 11: THE FINAL PUBLICATION NOTEBOOK.

Design principles:
  1. GITHUB-PULL: cells clone github.com/keshavkrishnan08/SDE and import the
     actual repo modules — the code that runs IS the code in the repo, not a
     rewritten copy embedded in the notebook.
  2. DUAL ARCHITECTURE: trains BOTH the closed-form (Mixture-of-OU) and the
     rollout (Euler-Maruyama latent rollout) models, plus their ENSEMBLE.
  3. CHAMPION SELECTION on validation (legitimate model selection, never test).
  4. ULTRA-ROBUST: every import is fallback-guarded, every stage is safe_stage
     wrapped, and every cell body has its own try/except — a failure anywhere
     prints and continues; the run always reaches the final zip.
  5. COMPLETE SUITE: SkyGPT exact benchmark (all variants), baselines, ablations,
     stratified+DM, CV, PIT/bootstrap, ramp, CTI, reliability, sampling, compute,
     Holm, economics+sensitivity, figures, LaTeX.
"""
import json, sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))
from _combined_generator import build_nb

REPO_HTTPS = "https://github.com/keshavkrishnan08/SDE.git"
REPO_ZIP   = "https://github.com/keshavkrishnan08/SDE/archive/refs/heads/main.zip"


# ============================================================
# Cell 1 — Environment setup (inline: runs before the repo exists locally)
# ============================================================
SETUP_CODE = '''\
# ==== Setup: environment, directories, torch warmup ====
import os, sys, json, math, time, gc, shutil, subprocess, traceback
from pathlib import Path
import numpy as np, pandas as pd

# torch._dynamo warmup (some Kaggle builds crash on lazy import at optimizer creation)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
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
    print("[WARN] No GPU — enable a GPU runtime. Training both architectures on CPU is impractical.")

ROOT = (Path("/kaggle/working") if IN_KAGGLE else Path.cwd()) / "final_run"
PERSIST_DIR = ROOT / "outputs"; WORK_DIR = ROOT / "work"; DATA_DIR = WORK_DIR / "data"
CHECKPOINT_DIR = PERSIST_DIR / "checkpoints"; RESULTS_DIR = PERSIST_DIR / "results"
LATENT_DIR = PERSIST_DIR / "latents"; SPLITS_DIR = PERSIST_DIR / "splits"
EXTENDED_DIR = PERSIST_DIR / "extended"; FIGURES_DIR = PERSIST_DIR / "figures"
for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LATENT_DIR, SPLITS_DIR, EXTENDED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===== Run configuration (the only knobs you may want to touch) =====
Z_DIM = 64
SKIPPD_VAE_EPOCHS  = 12     # CS-VAE epochs
CLOSEDFORM_EPOCHS  = 60     # closed-form SDE training epochs
ROLLOUT_EPOCHS     = 35     # rollout SDE epochs (each epoch costs ~3-4x closed-form)
CV_EPOCHS          = 15     # cross-validation epochs per fold
CV_MAX_FOLDS       = 6
print(f"PERSIST_DIR={PERSIST_DIR}")
print(f"Config: VAE={SKIPPD_VAE_EPOCHS}ep, closed-form={CLOSEDFORM_EPOCHS}ep, "
      f"rollout={ROLLOUT_EPOCHS}ep, CV={CV_EPOCHS}ep x {CV_MAX_FOLDS} folds")
'''


# ============================================================
# Cell 2 — Clone the repo + import every CODE constant (fallback-guarded)
# ============================================================
CLONE_AND_IMPORT_CODE = f'''\
# ==== Pull the SolarSDE codebase from GitHub and import the actual modules ====
# The code that runs below IS the repo code (github.com/keshavkrishnan08/SDE),
# not a copy embedded in this notebook.
REPO_HTTPS = "{REPO_HTTPS}"
REPO_ZIP   = "{REPO_ZIP}"
REPO_DIR = ROOT / "sde_repo"

def _clone_repo():
    if (REPO_DIR / "notebooks" / "_solarsde_v2.py").exists():
        print(f"  repo already present at {{REPO_DIR}}")
        # refresh to latest main (best effort)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
                       capture_output=True, timeout=120)
        return True
    for attempt in range(1, 4):
        try:
            print(f"  git clone (attempt {{attempt}}) ...")
            r = subprocess.run(["git", "clone", "--depth", "1", REPO_HTTPS, str(REPO_DIR)],
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and (REPO_DIR / "notebooks").exists():
                return True
            print(f"    clone failed: {{r.stderr[:200]}}")
        except Exception as e:
            print(f"    clone error: {{e}}")
        time.sleep(5)
    # Fallback: download the repo as a zip archive
    try:
        print("  falling back to zip archive download ...")
        import urllib.request, zipfile, io
        with urllib.request.urlopen(REPO_ZIP, timeout=300) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        zf.extractall(ROOT)
        extracted = next(ROOT.glob("SDE-*"))
        if REPO_DIR.exists(): shutil.rmtree(REPO_DIR)
        extracted.rename(REPO_DIR)
        return (REPO_DIR / "notebooks").exists()
    except Exception as e:
        print(f"    zip fallback failed: {{e}}")
        return False

if not _clone_repo():
    raise RuntimeError("Could not obtain the SolarSDE repo from GitHub — check network/repo access.")
MODULE_DIR = REPO_DIR / "notebooks"
sys.path.insert(0, str(MODULE_DIR))
print(f"  modules dir: {{MODULE_DIR}}")
print(f"  repo modules: {{sorted(p.name for p in MODULE_DIR.glob('_*.py'))}}")

# ---- Fallback-guarded imports: a missing/broken module never stops the run ----
def _safe_import(module, names):
    out = {{}}
    try:
        mod = __import__(module, fromlist=names)
        for n in names:
            out[n] = getattr(mod, n)
        print(f"  [OK]   {{module}}: {{len(names)}} constants")
    except Exception as e:
        print(f"  [FAIL] {{module}}: {{type(e).__name__}}: {{str(e)[:120]}}")
        for n in names:
            out[n] = f'print("[SKIP] {{n}} unavailable — module {{module}} failed to import")'
    return out

globals().update(_safe_import("_master_hardening", ["safe_stage"]))
globals().update(_safe_import("_combined_generator",
    ["SHARED_CODE", "BASELINES_CODE", "STRATIFIED_CODE", "ANALYSIS_CODE"]))
globals().update(_safe_import("_final_generator",
    ["LOAD_DATA_TOLERANT_CODE", "RAMP_AUROC_CODE", "BOOTSTRAP_CIS_CODE",
     "PIT_RELIABILITY_CODE", "ECONOMIC_CAISO_CODE", "LATEX_TABLES_CODE", "ZIP_DOWNLOAD_CODE"]))
globals().update(_safe_import("_colab_master_generator",
    ["CTI_VALIDATION_CODE", "HOLM_BONFERRONI_CODE"]))
globals().update(_safe_import("_skippd_pipeline",
    ["SKIPPD_DOWNLOAD_FULL_CODE", "SKIPPD_PREP_CODE", "SKIPPD_VAE_CODE",
     "SKIPPD_LATENTS_WRITE_CODE", "SKIPPD_HORIZON_OVERRIDE_CODE"]))
globals().update(_safe_import("_solarsde_v2",
    ["MDN_ARCHITECTURE_CODE", "STAGE_0_V2_CODE", "POST_STAGE0_V2_VERIFY_CODE", "ABLATIONS_V2_CODE"]))
globals().update(_safe_import("_solarsde_rollout",
    ["ROLLOUT_ARCH_CODE", "ABLATIONS_ROLLOUT_CODE"]))
globals().update(_safe_import("_ensemble_eval",
    ["STASH_CLOSEDFORM_CODE", "STASH_ROLLOUT_CODE", "CHAMPION_SELECT_CODE",
     "SKYGPT_TRIPLE_BENCHMARK_CODE"]))
globals().update(_safe_import("_skippd_extras",
    ["IMPLEMENTATION_DETAILS_CODE", "DATA_CARD_CODE", "COMPUTATIONAL_COST_CODE",
     "RELIABILITY_LEVELS_CODE", "SAMPLING_EFFICIENCY_CODE", "ECONOMIC_SENSITIVITY_CODE",
     "CROSS_VALIDATION_V2_CODE"]))

# If safe_stage itself failed to import, provide a minimal local fallback.
if isinstance(globals().get("safe_stage"), str):
    def safe_stage(name, code):
        ind = "\\n".join("    " + l if l else "" for l in code.splitlines())
        return (f"try:\\n{{ind}}\\nexcept Exception as _e:\\n"
                f"    import traceback; traceback.print_exc()\\n"
                f"    print('[STAGE FAILED] {{name}} — continuing.')\\n")
    print("  [WARN] using local fallback safe_stage")
print("\\nAll modules wired. Code provenance: github.com/keshavkrishnan08/SDE @ main")
'''


def _cell(stage_name: str, inner: str) -> str:
    """Cell body: exec(safe_stage(...)) with an outer try/except so even a
    NameError (missing import) or safe_stage failure can't stop Run-All."""
    return (
        f"# ==== {stage_name} ====\n"
        f"try:\n"
        f"    exec(safe_stage({stage_name!r}, {inner}), globals())\n"
        f"except Exception:\n"
        f"    import traceback; traceback.print_exc()\n"
        f"    print('[CELL FAILED] {stage_name} — continuing to next cell.')\n"
    )


def _cell_raw(stage_name: str, raw_stmt: str) -> str:
    """Cell that runs a raw statement (not a CODE constant) with protection."""
    return (
        f"# ==== {stage_name} ====\n"
        f"try:\n"
        + "".join(f"    {l}\n" for l in raw_stmt.splitlines())
        + f"except Exception:\n"
        f"    import traceback; traceback.print_exc()\n"
        f"    print('[CELL FAILED] {stage_name} — continuing to next cell.')\n"
    )


def nb_final():
    cells = [
        ("markdown",
         "# SolarSDE — Final Publication Notebook\n\n"
         "**Code provenance:** every stage below pulls and runs the actual modules from "
         "[github.com/keshavkrishnan08/SDE](https://github.com/keshavkrishnan08/SDE) — nothing is "
         "rewritten or embedded.\n\n"
         "**What this notebook produces (one GPU run):**\n"
         "1. Full SKIPP'D pipeline (517 days, VAE + optical-flow motion + CTI)\n"
         "2. **Both architectures** trained: closed-form Mixture-of-OU *and* Euler-Maruyama latent rollout\n"
         "3. **Ensemble** of the two + **champion selection on validation** (never test)\n"
         "4. SkyGPT exact-benchmark (identical Nov–Dec 2019 cloudy test) for **all variants**, full 1–30 min band\n"
         "5. Baselines, ablations, stratified+DM, leave-one-month-out CV, PIT/bootstrap, ramp AUROC, CTI "
         "validation, multi-level reliability, sampling efficiency, compute cost, Holm-Bonferroni, "
         "CAISO economics + sensitivity, figures, LaTeX tables\n\n"
         "Every stage is failure-isolated: an error prints and the run continues to the final zip.\n\n"
         "*Honesty note: the head-to-head vs SkyGPT (CRPS 2.81 at h=15, their cloudy test) is reported "
         "exactly as measured — whichever way it comes out.*"),

        ("markdown", "## 0. Environment"),
        ("code", SETUP_CODE),

        ("markdown", "## 1. Pull the codebase from GitHub (the repo code IS the experiment code)"),
        ("code", CLONE_AND_IMPORT_CODE),

        ("markdown", "## 2. Download all data — SKIPP'D (~2.3 GB) + SkyGPT exact test set"),
        ("code", _cell("DOWNLOAD_SKIPPD", "SKIPPD_DOWNLOAD_FULL_CODE")),

        ("markdown", "## 3. Preprocess — clear-sky-PV, ramps, chronological splits"),
        ("code", _cell("SKIPPD_PREP", "SKIPPD_PREP_CODE")),

        ("markdown", "## 4. CS-VAE (64×64 → 64-d) + encode all frames + optical-flow motion features"),
        ("code", _cell("SKIPPD_VAE", "SKIPPD_VAE_CODE")),

        ("markdown", "## 5. CTI + write the {splits, extended, latents} contract"),
        ("code", _cell("SKIPPD_WRITE", "SKIPPD_LATENTS_WRITE_CODE")),

        ("markdown", "## 6. Shared metrics + load tensors + 1-min horizon config"),
        ("code", _cell("SHARED", "SHARED_CODE")),
        ("code", _cell("LOAD_DATA", "LOAD_DATA_TOLERANT_CODE")),
        ("code", _cell("HORIZON_OVERRIDE", "SKIPPD_HORIZON_OVERRIDE_CODE")),

        ("markdown", "## 6a. Data card + implementation details (reproducibility)"),
        ("code", _cell("DATA_CARD", "DATA_CARD_CODE")),
        ("code", _cell("IMPLEMENTATION_DETAILS", "IMPLEMENTATION_DETAILS_CODE")),

        ("markdown", "## 7. ARCHITECTURE A — Closed-form Mixture-of-OU: train + calibrate + evaluate"),
        ("code", _cell("CLOSEDFORM_ARCH", "MDN_ARCHITECTURE_CODE")),
        ("code", _cell_raw("CLOSEDFORM_GLUE",
                           "# Keep a named reference to the closed-form class before the rollout\n"
                           "# architecture overwrites the TemporalLatentSDE alias.\n"
                           "ClosedFormSDE = TemporalLatentSDE\n"
                           "print('ClosedFormSDE alias saved.')")),
        ("code", _cell_raw("CLOSEDFORM_TRAIN",
                           "exec(safe_stage('STAGE0_CLOSEDFORM',\n"
                           "     STAGE_0_V2_CODE.replace('EPOCHS = 60', f'EPOCHS = {CLOSEDFORM_EPOCHS}')), globals())")),
        ("code", _cell("CLOSEDFORM_VERIFY", "POST_STAGE0_V2_VERIFY_CODE")),
        ("code", _cell("STASH_CLOSEDFORM", "STASH_CLOSEDFORM_CODE")),

        ("markdown", "## 8. ARCHITECTURE B — Euler-Maruyama latent rollout: train + calibrate + evaluate"),
        ("code", _cell("ROLLOUT_ARCH", "ROLLOUT_ARCH_CODE")),
        ("code", _cell_raw("ROLLOUT_TRAIN",
                           "exec(safe_stage('STAGE0_ROLLOUT',\n"
                           "     STAGE_0_V2_CODE.replace('EPOCHS = 60', f'EPOCHS = {ROLLOUT_EPOCHS}')), globals())")),
        ("code", _cell("ROLLOUT_VERIFY", "POST_STAGE0_V2_VERIFY_CODE")),
        ("code", _cell("STASH_ROLLOUT", "STASH_ROLLOUT_CODE")),

        ("markdown", "## 9. Champion selection (on VALIDATION) — closed-form vs rollout vs ensemble"),
        ("code", _cell("CHAMPION_SELECT", "CHAMPION_SELECT_CODE")),

        ("markdown", "## 10. SkyGPT EXACT BENCHMARK — all variants, identical cloudy test, full 1–30 min band"),
        ("code", _cell("SKYGPT_TRIPLE_BENCHMARK", "SKYGPT_TRIPLE_BENCHMARK_CODE")),

        ("markdown", "## 11. Baselines (persistence, smart-persistence, LSTM, MC-Dropout, CSDI)"),
        ("code", _cell("BASELINES", "BASELINES_CODE")),

        ("markdown", "## 12. Ablations (champion-matched: closed-form or rollout native)"),
        ("code", _cell_raw("ABLATIONS",
                           "_abl = ABLATIONS_V2_CODE if globals().get('CHAMPION_SINGLE', 'closedform') == 'closedform' \\\n"
                           "       else ABLATIONS_ROLLOUT_CODE\n"
                           "exec(safe_stage('ABLATIONS', _abl), globals())")),

        ("markdown", "## 13. Stratified eval + Diebold-Mariano significance"),
        ("code", _cell("STRATIFIED", "STRATIFIED_CODE")),

        ("markdown", "## 14. Leave-one-month-out cross-validation"),
        ("code", _cell("CROSS_VALIDATION", "CROSS_VALIDATION_V2_CODE")),

        ("markdown", "## 15. PIT / reliability + bootstrap CIs"),
        ("code", _cell("PIT_RELIABILITY", "PIT_RELIABILITY_CODE")),
        ("code", _cell("BOOTSTRAP_CIS", "BOOTSTRAP_CIS_CODE")),

        ("markdown", "## 16. Ramp AUROC + CTI physical validation"),
        ("code", _cell("RAMP_AUROC", "RAMP_AUROC_CODE")),
        ("code", _cell("CTI_VALIDATION", "CTI_VALIDATION_CODE")),

        ("markdown", "## 17. Multi-level reliability + sampling efficiency + computational cost"),
        ("code", _cell("RELIABILITY_LEVELS", "RELIABILITY_LEVELS_CODE")),
        ("code", _cell("SAMPLING_EFFICIENCY", "SAMPLING_EFFICIENCY_CODE")),
        ("code", _cell("COMPUTATIONAL_COST", "COMPUTATIONAL_COST_CODE")),

        ("markdown", "## 18. Holm-Bonferroni + CAISO economics + sensitivity"),
        ("code", _cell("HOLM_BONFERRONI", "HOLM_BONFERRONI_CODE")),
        ("code", _cell("ECONOMIC_CAISO", "ECONOMIC_CAISO_CODE")),
        ("code", _cell("ECONOMIC_SENSITIVITY", "ECONOMIC_SENSITIVITY_CODE")),

        ("markdown", "## 19. Analysis figures + LaTeX tables"),
        ("code", _cell("ANALYSIS", "ANALYSIS_CODE")),
        ("code", _cell("LATEX_TABLES", "LATEX_TABLES_CODE")),

        ("markdown", "## Final — Zip the complete paper package"),
        ("code", _cell("ZIP", "ZIP_DOWNLOAD_CODE")),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    path = NB_DIR / "11_final_publication.ipynb"
    nb = nb_final()
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path.name}: {path.stat().st_size / 1024:.1f} KB ({len(nb['cells'])} cells)")
