"""Build Notebook 08: Single master Colab notebook.

ONE notebook that runs the entire SolarSDE pipeline end-to-end for Energy Reports
submission. Designed for Colab (preferably Pro+ for 24h sessions, or Pro across
2 sessions). All artifacts persist to Google Drive so partial runs survive
session disconnects — re-run and finished stages skip.

What it produces (single Energy Reports submission package):
  - Golden CO probabilistic solar forecasts at h=1,5,10,20,30 min
  - 5-fold leave-one-day-out cross-validation across train days
  - Conformal calibration + PIT + reliability + sharpness
  - 5 baselines (persistence, smart-persistence, LSTM, MC-Dropout, CSDI)
  - 4 ablations (A2 no-CTI, A3 no-VAE, A4 no-score, A5 no-SDE)
  - Bootstrap CIs at all horizons + Diebold-Mariano + Holm-Bonferroni correction
  - Ramp event detection AUROC + CTI lead-time
  - CTI vs cloud-cover Spearman validation (physical-meaningfulness)
  - CAISO economic value ($/yr per GW)
  - 3 LaTeX tables ready to paste into the paper
  - 8 publication figures (PDF + PNG)

Estimated runtime: ~15-19 hours on Colab Pro+ (A100). Set DRIVE_PERSIST = True
at top of notebook to checkpoint to Drive so partial runs survive disconnects.
"""

import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

from _combined_generator import (
    build_nb,
    SHARED_CODE,
    STAGE_MINUS1_CODE, STAGE0_CODE,
    BASELINES_CODE, ABLATIONS_CODE,
    CALIBRATION_CODE, STRATIFIED_CODE, ANALYSIS_CODE,
)
from _master_hardening import (
    PREFLIGHT_SANITY_CODE, STAGE_M1_SAFE_FALLBACK_CODE,
    POST_STAGE0_VERIFY_CODE, safe_stage,
)
from _solarsde_v2 import (
    MDN_ARCHITECTURE_CODE, STAGE_0_V2_CODE, POST_STAGE0_V2_VERIFY_CODE,
)
from _generator import (
    CLOUDCV_DOWNLOAD, CLOUDCV_EXTRACT, BMS_DOWNLOAD, PREPROCESS_CODE,
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


HEADER_MASTER_MD = """# SolarSDE — Master Notebook for Energy Reports Submission

**Single end-to-end notebook.** Runs the complete pipeline: data, training, 5-fold CV, baselines, ablations, statistical pack, figures, LaTeX tables. Result: a downloadable zip with everything you need for the paper.

**Recommended environment:** Colab Pro+ (A100 GPU, 24h session). Colab Pro (T4, 12h) requires 2 sessions with Drive persistence.

## What this produces

| Section | Outputs |
|---|---|
| Data + VAE | Golden CO splits, 90-day extended BMS, Golden VAE checkpoint, latents + CTI + physics features |
| Main model | SolarSDE (CTI-gated Neural SDE + Score Decoder) trained on Golden |
| Validation | 5-fold leave-one-day-out cross-validation across train days |
| Baselines | Persistence, Smart-Persistence, LSTM, MC-Dropout, CSDI |
| Ablations | A2 (no-CTI), A3 (no-VAE/PCA), A4 (no-score), A5 (no-SDE/ODE) |
| Stats | Bootstrap CIs (B=1000) at all 5 horizons, Diebold-Mariano + Holm-Bonferroni |
| Calibration | Conformal, PIT histograms, reliability diagrams, sharpness |
| Operations | Ramp detection AUROC, CTI lead-time analysis, CAISO economic value ($/yr per GW) |
| Validation | CTI vs cloud-cover Spearman correlation (physical-meaningfulness check) |
| Paper-ready | 3 LaTeX tables + 8 PDF/PNG figures |
| Final | Single zip in /content/drive/MyDrive/ for download |

## Runtime by stage

| Stage | Time | Skips on rerun if |
|---|---|---|
| Setup + Drive mount | 1 min | always runs |
| Soft fast-start (pull cached) | 1-5 min | nothing to pull |
| Retrain Golden (conditional) | ~3-4h | vae_best.pt + latents exist |
| Image features | ~30 min | image_features.npy exists |
| Train SolarSDE seed 42 | ~3-4h | sde_best.pt + score_best.pt exist |
| 5-fold CV (~30 min × 5) | ~2.5h | cv_results.csv exists |
| Standard baselines | ~2-3h | each baseline CSV exists |
| Ablations A2/A3/A4/A5 | ~2h | each ablation CSV exists |
| Stage C+ corrected inference | ~30 min | per_horizon_preds/ populated |
| Calibration + Stratified + DM | ~1h | each stage's CSV exists |
| PIT + reliability + bootstrap | ~20 min | sharpness_summary.csv exists |
| Ramp AUROC + CTI lead-time | ~10 min | ramp_detection_auroc.csv exists |
| CTI vs cloud cover validation | ~10 min | cti_validation.csv exists |
| Holm-Bonferroni correction | <5 min | holm_bonferroni.csv exists |
| Economic value (CAISO) | ~15 min | economic_value_caiso.csv exists |
| Publication figures + LaTeX tables | ~10 min | always runs (cheap) |
| Final zip | ~5 min | always runs |

**Total fresh run: ~15-19 hours. Resume runs: ~30 min to 4h depending on state.**

## Setup checklist

1. Runtime → Change runtime type → **A100 GPU** (or T4 with 2-session plan)
2. Set `DRIVE_PERSIST = True` in the first code cell (Drive mount auto-prompts for auth)
3. Run All. If session disconnects: re-open, run all again — finished stages skip via Drive checkpoints
4. When done: download `solarsde_paper_package.zip` from `/content/drive/MyDrive/`
"""


# ================================================================
# COLAB-OPTIMIZED SETUP (Drive mount + persistence)
# ================================================================

SETUP_COLAB_CODE = '''\
# ==== Colab setup + Google Drive persistence ====
import os, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

# Set DRIVE_PERSIST=True (default) to checkpoint to Google Drive so partial
# runs survive Colab session disconnects. Set False for ephemeral runs.
DRIVE_PERSIST = True

if IN_COLAB and DRIVE_PERSIST:
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST_DIR = Path("/content/drive/MyDrive/solarsde")
    WORK_DIR = Path("/content/solarsde_work")
elif IN_COLAB:
    PERSIST_DIR = Path("/content/solarsde")
    WORK_DIR = Path("/content/solarsde_work")
elif IN_KAGGLE:
    PERSIST_DIR = Path("/kaggle/working/solarsde_outputs")
    WORK_DIR = Path("/kaggle/working/solarsde")
else:
    PERSIST_DIR = Path.cwd() / "solarsde_outputs"
    WORK_DIR = Path.cwd() / "solarsde_work"

for d in [PERSIST_DIR, WORK_DIR,
          PERSIST_DIR / "checkpoints", PERSIST_DIR / "results",
          PERSIST_DIR / "latents",     PERSIST_DIR / "splits",
          PERSIST_DIR / "extended",    PERSIST_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

DATA_DIR        = WORK_DIR / "data"
CHECKPOINT_DIR  = PERSIST_DIR / "checkpoints"
RESULTS_DIR     = PERSIST_DIR / "results"
LATENT_DIR      = PERSIST_DIR / "latents"
SPLITS_DIR      = PERSIST_DIR / "splits"
EXTENDED_DIR    = PERSIST_DIR / "extended"
FIGURES_DIR     = PERSIST_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'local'}")
print(f"PERSIST_DIR: {PERSIST_DIR}  (DRIVE_PERSIST={DRIVE_PERSIST})")
print(f"WORK_DIR:    {WORK_DIR}")

# Install dependencies
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
    print(f"  GPU: {torch.cuda.get_device_name(0)}  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
'''


# Reuse the soft fast-start from _final_generator (try to pull cached artifacts
# from GitHub; never crash if missing — retrain stage will produce them)
FAST_START_GITHUB_CODE = '''\
# ==== Soft fast-start: try to pull cached artifacts from GitHub ====
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

print("Trying GitHub cache (best-effort) ...")
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
    if gh_pull_soft(rel, dest):
        n_have += 1
print(f"  artifacts available: {n_have}")

HAVE_VAE     = (CHECKPOINT_DIR / "vae_best.pt").exists()
HAVE_SPLITS  = (SPLITS_DIR / "train.parquet").exists()
HAVE_LATENTS = all((LATENT_DIR / f"{s}_latents.npy").exists() for s in ["train", "val", "test"])
HAVE_KT      = all((LATENT_DIR / f"{s}_kt.npy").exists() for s in ["train", "val", "test"])
HAVE_PHYS    = all((LATENT_DIR / f"{s}_physics_features.npy").exists() for s in ["train", "val", "test"])
HAVE_EXTENDED = (EXTENDED_DIR / "train.parquet").exists()
HAVE_EXT      = HAVE_EXTENDED   # alias for backward compat with LOAD_DATA_TOLERANT_CODE
NEED_GOLDEN_RETRAIN = not (HAVE_VAE and HAVE_SPLITS and HAVE_LATENTS and HAVE_KT and HAVE_PHYS)
print(f"NEED_GOLDEN_RETRAIN = {NEED_GOLDEN_RETRAIN}")

def _have_real_image_features():
    import numpy as _np
    for _s in ["train", "val", "test"]:
        _p = LATENT_DIR / f"{_s}_image_features.npy"
        if not _p.exists(): return False
        _a = _np.load(_p, mmap_mode="r")
        if _a.size == 0 or not _a.any(): return False
    return True
NEED_IMAGE_FEATURES = not _have_real_image_features()
print(f"NEED_IMAGE_FEATURES = {NEED_IMAGE_FEATURES}")

# ==== Stage-0 retraining gate (consumed by STAGE0_CODE) ====
SDE_CKPT   = CHECKPOINT_DIR / "sde_best.pt"
SCORE_CKPT = CHECKPOINT_DIR / "score_best.pt"
NEED_NB2_TRAINING = not (SDE_CKPT.exists() and SCORE_CKPT.exists())
print(f"NEED_NB2_TRAINING   = {NEED_NB2_TRAINING}  (SDE+Score will be trained inline if True)")
'''


# ================================================================
# K-FOLD LEAVE-ONE-DAY-OUT CROSS VALIDATION
# ================================================================

K_FOLD_CV_CODE = '''\
# ==== Leave-one-day-out cross-validation across train days ====
# Strongest reviewer answer to "only 5 days?!" — show generalization across
# day-level holdouts. We share the VAE across folds (unsupervised, no label
# leakage) and retrain only the SDE + Score Decoder per fold.
#
# Per fold: ~30-40 min on A100, ~1.5h on T4. 5 folds = ~2.5-7.5 hours total.

CV_OUT = RESULTS_DIR / "cv_results.csv"
if CV_OUT.exists():
    print(f"[SKIP] CV already done -> {CV_OUT}")
    cv_summary = pd.read_csv(CV_OUT)
    print(cv_summary.to_string(index=False))
else:
    print("=" * 70)
    print("LEAVE-ONE-DAY-OUT CROSS VALIDATION")
    print("=" * 70)

    # Identify unique training days
    tr_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    if "image_exists" in tr_df.columns:
        tr_df = tr_df[tr_df["image_exists"]].reset_index(drop=True)
    tr_df["date"] = pd.to_datetime(tr_df["timestamp"]).dt.date
    days = sorted(tr_df["date"].unique())
    print(f"Train days: {[str(d) for d in days]}")

    # Indices of each day in the full train tensor arrays
    day_idx = {d: tr_df.index[tr_df["date"] == d].tolist() for d in days}

    # Pull from the already-loaded `data` dict so the cov dim matches what
    # STAGE 0 was trained with. (Re-reading from disk + concat'ing image
    # features can drift if STAGE_M1_SAFE_FALLBACK wrote zero-fill features
    # AFTER LOAD_DATA had already set C_DIM.)
    z_all   = data["train"]["Z"]
    cti_all = data["train"]["cti"]
    kt_all  = data["train"]["kt"]
    cov_all = data["train"]["cov"]
    ghi_all = data["train"]["ghi"]
    gcs_all = data["train"]["gcs"]
    c_dim_fold = cov_all.shape[1]
    print(f"  CV cov dim: {c_dim_fold}  (matches C_DIM={C_DIM} from LOAD_DATA: {c_dim_fold == C_DIM})")

    fold_rows = []
    for fold_i, holdout_day in enumerate(days):
        print(f"\\n--- Fold {fold_i + 1}/{len(days)}: holding out {holdout_day} ---")
        tr_mask = np.zeros(len(z_all), dtype=bool)
        for d in days:
            if d != holdout_day:
                for i in day_idx[d]: tr_mask[i] = True
        te_mask = ~tr_mask

        z_tr_fold, z_te_fold = z_all[tr_mask], z_all[te_mask]
        cti_tr_fold, cti_te_fold = cti_all[tr_mask], cti_all[te_mask]
        kt_tr_fold, kt_te_fold = kt_all[tr_mask], kt_all[te_mask]
        cov_tr_fold, cov_te_fold = cov_all[tr_mask], cov_all[te_mask]
        ghi_te_fold = ghi_all[te_mask]
        gcs_te_fold = gcs_all[te_mask]

        print(f"  train={len(z_tr_fold)}  test (held-out day)={len(z_te_fold)}")
        if len(z_te_fold) < 200:
            print(f"  [SKIP] held-out day has too few samples")
            continue

        # Train SDE on this fold (reduced epochs for speed)
        torch.manual_seed(42 + fold_i)
        np.random.seed(42 + fold_i)

        class MHDS_CV(Dataset):
            def __init__(self, z, cti, c, hs=(1, 5, 10, 30, 60, 90, 120, 180), seed=42):
                self.z = z; self.cti = cti; self.c = c
                self.hs = hs; self.rng = np.random.RandomState(seed)
                self.maxh = max(hs); self.idx = np.arange(len(z) - self.maxh)
            def __len__(self): return len(self.idx)
            def __getitem__(self, i):
                ii = self.idx[i]; k = int(self.rng.choice(self.hs))
                return {"z_t": torch.from_numpy(self.z[ii]),
                        "z_next": torch.from_numpy(self.z[ii + k]),
                        "k": torch.tensor(k, dtype=torch.float32),
                        "cti_t": torch.tensor(self.cti[ii], dtype=torch.float32),
                        "c_t": torch.from_numpy(self.c[ii])}
        mh = MHDS_CV(z_tr_fold, cti_tr_fold, cov_tr_fold, seed=42 + fold_i)
        dl = DataLoader(mh, batch_size=512, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
        sde_cv = LatentNeuralSDE(z_dim=Z_DIM, c_dim=c_dim_fold).to(DEVICE)
        opt = torch.optim.Adam(sde_cv.parameters(), lr=5e-4)
        for ep in range(1, 21):   # 20 epochs (vs 30 for main model)
            sde_cv.train(); tl = 0; n = 0
            for b in dl:
                z = b["z_t"].to(DEVICE); zn = b["z_next"].to(DEVICE)
                k = b["k"].float().unsqueeze(-1).to(DEVICE); t = k / 180.0
                cti = b["cti_t"].unsqueeze(-1).to(DEVICE); c = b["c_t"].to(DEVICE)
                mu = sde_cv.drift(z, t, c); sigma = sde_cv.diffusion(z, cti)
                dz = (zn - z) / k
                drift_l = F.mse_loss(mu, dz)
                resid = zn - z - mu * k
                tv = (resid ** 2) / k.clamp(min=1.0)
                sq = sigma.pow(2).clamp(min=1e-6)
                diff_l = F.mse_loss(torch.log(sq + 1e-8), torch.log(tv + 1e-8))
                loss = drift_l + 0.5 * diff_l
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sde_cv.parameters(), 1.0); opt.step()
                tl += loss.item(); n += 1
            if ep % 5 == 0: print(f"    SDE fold{fold_i+1} ep {ep}/20: loss={tl/n:.4f}")

        # Train score decoder
        class SDS_CV(Dataset):
            def __init__(self, z, cti, c, kt, hs=(1, 5, 10, 30, 60, 90, 120, 180), seed=42):
                self.z = z; self.cti = cti; self.c = c; self.kt = kt
                self.hs = hs; self.rng = np.random.RandomState(seed); self.maxh = max(hs)
            def __len__(self): return len(self.z) - self.maxh
            def __getitem__(self, i):
                k = int(self.rng.choice(self.hs))
                return {"kt_target": torch.tensor(self.kt[i + k], dtype=torch.float32),
                        "kt_current": torch.tensor(self.kt[i], dtype=torch.float32),
                        "z_t": torch.from_numpy(self.z[i]),
                        "cti_t": torch.tensor(self.cti[i], dtype=torch.float32),
                        "c_t": torch.from_numpy(self.c[i])}
        score_cv = CondScoreDecoder(z_dim=Z_DIM, c_dim=c_dim_fold, predict_mode='delta').to(DEVICE)
        opt2 = torch.optim.Adam(score_cv.parameters(), lr=1e-4)
        sds = SDS_CV(z_tr_fold, cti_tr_fold, cov_tr_fold, kt_tr_fold, seed=42 + fold_i)
        sdl = DataLoader(sds, batch_size=512, shuffle=True, num_workers=2,
                         pin_memory=True, drop_last=True)
        for ep in range(1, 21):
            score_cv.train(); tl = 0; n = 0
            for b in sdl:
                loss_d = score_cv.training_loss(
                    b["kt_target"].unsqueeze(-1).to(DEVICE),
                    b["kt_current"].unsqueeze(-1).to(DEVICE),
                    b["z_t"].to(DEVICE), b["cti_t"].unsqueeze(-1).to(DEVICE),
                    b["c_t"].to(DEVICE))
                loss = loss_d["loss"] if isinstance(loss_d, dict) else loss_d
                opt2.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(score_cv.parameters(), 1.0); opt2.step()
                tl += loss.item(); n += 1
            if ep % 5 == 0: print(f"    Score fold{fold_i+1} ep {ep}/20: loss={tl/n:.4f}")

        # Eval on held-out day
        sde_cv.eval(); score_cv.eval()
        for h in HORIZONS:
            preds_l, truths_l = [], []
            n_eval_fold = len(z_te_fold) - h - 1
            for i in range(0, n_eval_fold, 32):
                end = min(i + 32, n_eval_fold); bs = end - i
                z0 = torch.from_numpy(z_te_fold[i:end]).to(DEVICE)
                z0 = z0.unsqueeze(1).repeat(1, N_SAMPLES, 1).reshape(-1, Z_DIM)
                cti0 = torch.from_numpy(cti_te_fold[i:end]).unsqueeze(-1).to(DEVICE)
                cti0 = cti0.unsqueeze(1).repeat(1, N_SAMPLES, 1).reshape(-1, 1)
                c0 = torch.from_numpy(cov_te_fold[i:end]).to(DEVICE)
                c0 = c0.unsqueeze(1).repeat(1, N_SAMPLES, 1).reshape(-1, c_dim_fold)
                kt0 = torch.from_numpy(kt_te_fold[i:end]).unsqueeze(-1).to(DEVICE)
                kt0 = kt0.unsqueeze(1).repeat(1, N_SAMPLES, 1).reshape(-1, 1)
                with torch.no_grad():
                    z = z0
                    # Use the clamped em_step from SHARED_CODE (drift/sigma/z
                    # bounded by Z_MEAN±8·Z_STD) — same path STAGE 0 inference
                    # uses. Without clamping, long-horizon (h>=10min = 60+
                    # Euler steps) rollouts drift OOD and PICP collapses.
                    for s in range(h):
                        t_norm = torch.full((bs * N_SAMPLES, 1),
                                            (s + 1) / 180.0, device=DEVICE)
                        z = em_step(sde_cv.drift, sde_cv.diffusion,
                                    z, t_norm, c0, cti0, 1.0)
                    kt_pred = score_cv.sample(z, cti0, c0, kt0, n=1).squeeze(-1).cpu().numpy()
                    kt_pred = kt_pred.reshape(bs, N_SAMPLES)
                # Guard against any residual NaN/Inf so one bad batch can't
                # poison the whole fold's metrics
                if not np.isfinite(kt_pred).all():
                    bad = (~np.isfinite(kt_pred)).sum()
                    kt_pred = np.nan_to_num(kt_pred, nan=1.0, posinf=2.0, neginf=0.0)
                    print(f"      [WARN] replaced {bad} non-finite kt samples in batch {i}")
                ghi_pred = kt_pred * gcs_te_fold[i:end][:, None]
                preds_l.append(ghi_pred); truths_l.append(ghi_te_fold[i + h:end + h])
            preds = np.concatenate(preds_l, axis=0); yt = np.concatenate(truths_l)
            crps = float(crps_empirical(yt, preds).mean())
            rmse = float(np.sqrt(((preds.mean(1) - yt) ** 2).mean()))
            picp = float(((np.percentile(preds, 5, axis=1) <= yt) &
                          (yt <= np.percentile(preds, 95, axis=1))).mean())
            fold_rows.append({"fold": fold_i + 1, "holdout_day": str(holdout_day),
                              "horizon_min": HORIZON_MIN[h], "crps": crps,
                              "rmse": rmse, "picp": picp, "n_eval": len(yt)})
            print(f"    h={HORIZON_MIN[h]:2d}min: CRPS={crps:.2f} RMSE={rmse:.2f} PICP={picp:.3f}")
        del sde_cv, score_cv, mh, dl, sds, sdl; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # Incremental save so a Kaggle session timeout mid-fold-5 doesn't
        # lose folds 1-4. cv_results.csv is finalized after the loop.
        pd.DataFrame(fold_rows).to_csv(
            RESULTS_DIR / "cv_results_per_fold.csv", index=False)

    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_results_per_fold.csv", index=False)

    # Aggregate: mean ± std across folds per horizon
    cv_summary = cv_df.groupby("horizon_min").agg(
        crps_mean=("crps", "mean"), crps_std=("crps", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        picp_mean=("picp", "mean"), picp_std=("picp", "std"),
        n_folds=("fold", "count"),
    ).reset_index()
    cv_summary.to_csv(CV_OUT, index=False)
    print(f"\\n5-fold CV summary (mean ± std across folds):")
    for _, r in cv_summary.iterrows():
        print(f"  h={int(r['horizon_min']):2d}min: CRPS = {r['crps_mean']:.2f} ± {r['crps_std']:.2f}, "
              f"RMSE = {r['rmse_mean']:.2f} ± {r['rmse_std']:.2f}, "
              f"PICP = {r['picp_mean']:.3f} ± {r['picp_std']:.3f}")
'''


# ================================================================
# CTI vs CLOUD COVER VALIDATION (physical-meaningfulness)
# ================================================================

CTI_VALIDATION_CODE = '''\
# ==== CTI vs cloud-cover Spearman correlation (physical-meaningfulness) ====
# Reviewer-required validation: prove that the learned CTI scalar correlates
# with a physically-measurable cloud variability indicator. We use rolling
# 5-minute std of GHI as a proxy for cloud cover variability (TSI-880 not
# always available in our dataset slice).

from scipy import stats

CTI_VAL_OUT = RESULTS_DIR / "cti_validation.csv"
if CTI_VAL_OUT.exists():
    print(f"[SKIP] CTI validation already done -> {CTI_VAL_OUT}")
else:
    rows = []
    for split in ["train", "val", "test"]:
        cti = np.load(LATENT_DIR / f"{split}_cti.npy")
        ghi = np.load(LATENT_DIR / f"{split}_ghi.npy")

        # Rolling GHI std as cloud-variability proxy
        W = 30   # 30 × 10s = 5 minutes
        ghi_std = np.zeros_like(ghi)
        for i in range(W, len(ghi)):
            ghi_std[i] = float(np.std(ghi[i - W:i]))

        # Only consider daytime points (GHI > 50 W/m²)
        valid = (ghi > 50) & (cti > 0)
        if valid.sum() < 100:
            continue

        rho_sp, p_sp = stats.spearmanr(cti[valid], ghi_std[valid])
        rho_pe, p_pe = stats.pearsonr(cti[valid], ghi_std[valid])

        # Additionally: cloud-fraction proxy using kt deviation
        kt = np.load(LATENT_DIR / f"{split}_kt.npy")
        kt_dev = np.abs(kt - 1.0)   # large deviation from 1 = either cloudy (low) or enhanced (high)
        rho_kt, p_kt = stats.spearmanr(cti[valid], kt_dev[valid])

        rows.append({
            "split": split, "n": int(valid.sum()),
            "spearman_cti_ghi_std": rho_sp, "p_spearman_ghi_std": p_sp,
            "pearson_cti_ghi_std": rho_pe, "p_pearson_ghi_std": p_pe,
            "spearman_cti_kt_dev": rho_kt, "p_spearman_kt_dev": p_kt,
        })
        print(f"  {split}: n={valid.sum()}, Spearman(CTI, rolling-GHI-std) = {rho_sp:.3f} (p={p_sp:.2e})")
        print(f"           Spearman(CTI, |kt-1|) = {rho_kt:.3f} (p={p_kt:.2e})")

    if rows:
        pd.DataFrame(rows).to_csv(CTI_VAL_OUT, index=False)
        # Interpretation
        max_sp = max(r["spearman_cti_ghi_std"] for r in rows)
        if max_sp > 0.5:
            print(f"\\n  [OK] CTI shows strong positive correlation with cloud variability (max ρ = {max_sp:.3f}).")
            print("       This validates CTI as a physically-meaningful scalar.")
        elif max_sp > 0.3:
            print(f"\\n  [MODERATE] CTI correlation with cloud variability is moderate (max ρ = {max_sp:.3f}).")
            print("           Still defensible — caveat that CTI captures latent dynamics, not direct cloud-cover.")
        else:
            print(f"\\n  [WEAK] CTI-cloud correlation is weak (max ρ = {max_sp:.3f}).")
            print("        Paper should discuss why latent dynamics may not align with direct cloud variability.")
'''


# ================================================================
# HOLM-BONFERRONI MULTIPLE-COMPARISON CORRECTION
# ================================================================

HOLM_BONFERRONI_CODE = '''\
# ==== Holm-Bonferroni correction on Diebold-Mariano p-values ====
# When we test SolarSDE vs multiple baselines at multiple horizons, we run
# many DM tests. Without correction, the chance of a false positive grows
# with the number of tests. Holm-Bonferroni controls family-wise error rate
# while being less conservative than plain Bonferroni.

HB_OUT = RESULTS_DIR / "holm_bonferroni_corrected.csv"
strat_p = RESULTS_DIR / "stratified_results.csv"
if not strat_p.exists():
    print(f"[SKIP] {strat_p.name} not found — Holm-Bonferroni needs DM p-values from stratified stage.")
elif HB_OUT.exists():
    print(f"[SKIP] Holm-Bonferroni done -> {HB_OUT}")
else:
    df = pd.read_csv(strat_p)
    p_cols = [c for c in df.columns if "p_value" in c.lower() or "pval" in c.lower() or "dm_p" in c.lower()]
    if not p_cols:
        print("[INFO] No DM p-value columns found in stratified_results.csv — skipping Holm-Bonferroni.")
    else:
        # Collect all p-values into one flat list
        from itertools import chain
        all_p = []
        for col in p_cols:
            for v in df[col].values:
                try:
                    fv = float(v)
                    if not np.isnan(fv): all_p.append((col, fv))
                except: pass
        if len(all_p) < 2:
            print("[INFO] Fewer than 2 valid p-values — Holm-Bonferroni not applicable.")
        else:
            # Holm-Bonferroni: sort ascending, threshold = alpha / (n - i + 1)
            ALPHA = 0.05
            all_p_sorted = sorted(all_p, key=lambda x: x[1])
            n = len(all_p_sorted)
            corrected = []
            reject_so_far = True
            for i, (col, p) in enumerate(all_p_sorted):
                thresh = ALPHA / (n - i)
                adj_p = min(1.0, p * (n - i))
                reject = (p <= thresh) and reject_so_far
                if not reject: reject_so_far = False
                corrected.append({"comparison": col, "raw_p": p, "rank": i + 1,
                                  "threshold_holm": thresh, "adjusted_p": adj_p,
                                  "reject_h0": reject})
            hb_df = pd.DataFrame(corrected)
            hb_df.to_csv(HB_OUT, index=False)
            n_reject = sum(r["reject_h0"] for r in corrected)
            print(f"Holm-Bonferroni correction (α=0.05, {n} tests):")
            print(f"  {n_reject}/{n} comparisons remain significant after correction.")
            print(hb_df.head(15).to_string(index=False))
'''


# ================================================================
# MASTER NOTEBOOK BUILDER
# ================================================================

def master_nb():
    """Build the single end-to-end Colab notebook."""
    cells = [
        ("markdown", HEADER_MASTER_MD),
        ("markdown", "## 0. Setup (Colab + Drive)"),
        ("code", SETUP_COLAB_CODE),
        ("code", FAST_START_GITHUB_CODE),
        ("markdown", "## 0a. Preflight sanity check (catches missing names + corrupt ckpts early)"),
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
        # Fallback runs BEFORE LOAD_DATA so cov dim is consistent from the
        # very first load — and any stale zero-fill that would mismatch an
        # existing SDE ckpt's c_dim is deleted up front.
        ("code", STAGE_M1_SAFE_FALLBACK_CODE),
        ("code", STAGE_MINUS1_CODE),

        ("markdown", "## 2. Load data tensors"),
        ("code", LOAD_DATA_TOLERANT_CODE),

        ("markdown", "## STAGE C — Train SolarSDE v2 (Persistence-Residual MDN)"),
        ("code", MDN_ARCHITECTURE_CODE),
        ("code", STAGE_0_V2_CODE),
        ("code", safe_stage("POST_STAGE0_V2_VERIFY", POST_STAGE0_V2_VERIFY_CODE)),

        ("markdown", "## STAGE CV — Leave-one-day-out cross-validation (reviewer requirement)"),
        ("code", safe_stage("K_FOLD_CV", K_FOLD_CV_CODE)),

        ("markdown", "## STAGE C+ — Corrected inference (advance time-deterministic covariates)"),
        ("code", safe_stage("CORRECTED_INFERENCE", CORRECTED_INFERENCE_CODE)),

        ("markdown", "## STAGE D — Standard baselines (persistence, smart-pers, LSTM, MC-Dropout, CSDI)"),
        ("code", safe_stage("BASELINES", BASELINES_CODE)),

        ("markdown", "## STAGE F — Ablations A2 (no-CTI), A4 (no-score), A5 (no-SDE), A3 (no-VAE PCA)"),
        ("code", safe_stage("ABLATIONS", ABLATIONS_CODE)),
        ("code", safe_stage("EXTRA_ABLATIONS", EXTRA_ABLATIONS_CODE)),

        ("markdown", "## STAGE G — Conformal calibration"),
        ("code", safe_stage("CALIBRATION", CALIBRATION_CODE)),

        ("markdown", "## STAGE H — Stratified eval + Diebold-Mariano test"),
        ("code", safe_stage("STRATIFIED", STRATIFIED_CODE)),

        ("markdown", "## STAGE I — PIT + reliability + sharpness + bootstrap CIs (all horizons)"),
        ("code", safe_stage("PIT_RELIABILITY", PIT_RELIABILITY_CODE)),
        ("code", safe_stage("BOOTSTRAP_CIS", BOOTSTRAP_CIS_CODE)),

        ("markdown", "## STAGE I+ — Ramp AUROC + CTI lead-time"),
        ("code", safe_stage("RAMP_AUROC", RAMP_AUROC_CODE)),

        ("markdown", "## STAGE I++ — CTI vs cloud-cover validation (physical-meaningfulness)"),
        ("code", safe_stage("CTI_VALIDATION", CTI_VALIDATION_CODE)),

        ("markdown", "## STAGE I+++ — Holm-Bonferroni multiple-comparison correction"),
        ("code", safe_stage("HOLM_BONFERRONI", HOLM_BONFERRONI_CODE)),

        ("markdown", "## STAGE K — Economic value (CAISO reserve simulation)"),
        ("code", safe_stage("ECONOMIC_CAISO", ECONOMIC_CAISO_CODE)),

        ("markdown", "## STAGE M — Analysis figures (CTI, regime, forecast traces)"),
        ("code", safe_stage("ANALYSIS", ANALYSIS_CODE)),

        ("markdown", "## STAGE N — LaTeX tables (3 tables, paper-ready)"),
        ("code", safe_stage("LATEX_TABLES", LATEX_TABLES_CODE)),

        ("markdown", "## Final — Zip everything to /content/drive/MyDrive/ for download"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    path = NB_DIR / "08_solarsde_master_colab.ipynb"
    nb = master_nb()
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path.name}: {path.stat().st_size / 1024:.1f} KB ({len(nb['cells'])} cells)")
