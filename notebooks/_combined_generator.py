"""Build the combined unattended notebook: Baselines + Ablations + Analysis + Figures.

Target: runs 4-6 hours on Kaggle P100 / Colab T4 GPU, fully unattended.
Produces all paper-ready tables, calibration, figures, and a final zip.

Design principles:
  - Each stage writes its outputs immediately to persistent storage
  - Each stage checks if its outputs already exist and skips if resuming
  - Comprehensive print-based progress (every major step logged)
  - Memory cleanup between stages (del + torch.cuda.empty_cache())
  - Defensive: all tensor ops bounded; indices checked; metrics handle edge cases
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def build_nb(cells):
    nb_cells = []
    for cell_type, src in cells:
        if isinstance(src, str):
            source = src.splitlines(keepends=True)
        else:
            source = src
        cell = {"cell_type": cell_type, "metadata": {}, "source": source}
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


# ================================================================
# Setup fragments (shared with per-stage notebooks, but consolidated)
# ================================================================

HEADER_MD = """# SolarSDE Combined Notebook — Baselines + Ablations + Analysis + Figures

**Runtime: ~4-6 hours on Kaggle P100 / Colab T4 GPU.** Designed for unattended runs.

**What this notebook does:**
1. Pulls Notebook 1 + Notebook 2 outputs from GitHub (trained VAE, SDE, Score Decoder, latents, splits)
2. Trains 5 baselines (Persistence, Smart Persistence, LSTM, MC-Dropout, CSDI) on 90-day BMS, evaluates on image test set
3. Runs 3 ablations (no-CTI, no-score, no-SDE-deterministic)
4. Applies post-hoc conformal calibration
5. Runs CTI analysis + regime clustering + economic value simulation
6. Generates all paper figures (CRPS vs horizon, reliability diagram, CTI scatter, economic value, PIT histogram)
7. Zips everything for download

**Resume-safe:** each major section checks if its output files already exist and skips.
If your session disconnects, just re-run all cells — already-completed stages will be skipped.

**Kaggle tips:** enable P100 GPU, commit the notebook periodically, free tier is 30hrs/week.
**Colab tips:** free T4 disconnects on idle; Colab Pro recommended for unattended runs.
"""

SETUP_CODE = '''\
# ==== Dependencies ====
import subprocess, sys
def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
pip_install("pvlib", "properscoring", "pyarrow", "tqdm")

# ==== Environment detection ====
import os, time, json, shutil, gc, math
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle")
print(f"Environment: {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'Local'}")

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount("/content/drive", force_remount=False)
    except Exception as e:
        print(f"Drive mount issue: {e}")
    PERSIST_DIR = Path("/content/drive/MyDrive/solarsde_outputs")
    WORK_DIR = Path("/content/solarsde")
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

print(f"Persistent storage: {PERSIST_DIR}")

# ==== GPU setup ====
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
    print("WARNING: CPU only. This notebook needs a GPU; please enable one.")
print(f"Using device: {DEVICE}")
'''

FAST_START_CODE = '''\
# ==== Fast-start: pull all required upstream artifacts from GitHub ====
# Notebook 1 outputs (VAE + latents + splits + extended) are on GitHub.
# Notebook 2 outputs (SDE + Score + main_results) should be in persistent storage
# from your recent Colab run; if missing, we attempt to pull anyway.

import requests
GITHUB_RAW = "https://raw.githubusercontent.com/keshavkrishnan08/SDE/main"

def gh_pull(rel_path, dest):
    """Fetch a single file from GitHub raw. Returns True on success."""
    url = f"{GITHUB_RAW}/{rel_path}"
    try:
        r = requests.get(url, timeout=180)
        if r.status_code == 200 and len(r.content) > 100:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True
        return False
    except Exception as e:
        print(f"  FAIL {rel_path}: {e}")
        return False

print("Checking persistent storage for upstream artifacts ...")

# Required files and their GitHub paths
required = {
    # Notebook 1 artifacts
    CHECKPOINT_DIR / "vae_best.pt":         "colab_outputs/checkpoints/vae_best.pt",
    SPLITS_DIR    / "train.parquet":        "colab_outputs/splits/train.parquet",
    SPLITS_DIR    / "val.parquet":          "colab_outputs/splits/val.parquet",
    SPLITS_DIR    / "test.parquet":         "colab_outputs/splits/test.parquet",
    EXTENDED_DIR  / "train.parquet":        "colab_outputs/extended/train.parquet",
    EXTENDED_DIR  / "val.parquet":          "colab_outputs/extended/val.parquet",
    EXTENDED_DIR  / "test.parquet":         "colab_outputs/extended/test.parquet",
}
for split in ["train", "val", "test"]:
    for key in ["latents", "cti", "ghi", "covariates", "is_ramp"]:
        required[LATENT_DIR / f"{split}_{key}.npy"] = f"colab_outputs/latents/{split}_{key}.npy"

# Notebook 2 artifacts (will only exist on GitHub if re-pushed; otherwise must be in Drive)
optional_nb2 = {
    CHECKPOINT_DIR / "sde_best.pt":    "colab_outputs/checkpoints/sde_best.pt",
    CHECKPOINT_DIR / "score_best.pt":  "colab_outputs/checkpoints/score_best.pt",
    RESULTS_DIR    / "solar_sde_main_results.csv": "colab_outputs/results/solar_sde_main_results.csv",
}

n_missing = 0
for dest, rel in required.items():
    if dest.exists() and dest.stat().st_size > 0:
        continue
    ok = gh_pull(rel, dest)
    if ok:
        print(f"  pulled {dest.name}  ({dest.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  MISSING: {rel}")
        n_missing += 1

print(f"\\nAttempting optional Notebook 2 artifacts ...")
for dest, rel in optional_nb2.items():
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} (from previous run)")
        continue
    ok = gh_pull(rel, dest)
    if ok:
        print(f"  pulled {dest.name}  ({dest.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  NOT on GitHub: {rel}")

# Hard requirement: VAE + latents
if n_missing > 0:
    raise RuntimeError(f"{n_missing} required Notebook 1 files missing — cannot proceed.")

# Notebook 2 artifacts may be in Drive from your recent run, not GitHub.
SDE_CKPT = CHECKPOINT_DIR / "sde_best.pt"
SCORE_CKPT = CHECKPOINT_DIR / "score_best.pt"
assert SDE_CKPT.exists() and SCORE_CKPT.exists(), (
    "Notebook 2 outputs (sde_best.pt, score_best.pt) not found in persistent storage. "
    "Please run Notebook 2 first, or ensure /content/drive/MyDrive/solarsde_outputs/checkpoints/ "
    "contains both files."
)
print(f"\\nAll required files present. Ready to run.")
'''

SHARED_CODE = '''\
# ==== Shared model definitions (matches Notebooks 1 + 2) ====

# --- CS-VAE (needed only for sanity; not retrained here) ---
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=64, channels=(32, 64, 128, 256)):
        super().__init__()
        layers, in_ch = [], 3
        for ch in channels:
            layers.extend([nn.Conv2d(in_ch, ch, 4, 2, 1),
                           nn.GroupNorm(min(32, ch), ch),
                           nn.SiLU(inplace=True)])
            in_ch = ch
        self.conv = nn.Sequential(*layers); self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(channels[-1], latent_dim)
        self.fc_lv = nn.Linear(channels[-1], latent_dim)
    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

# --- Neural SDE ---
class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(inplace=True), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)

class DriftNet(nn.Module):
    def __init__(self, z_dim=64, c_dim=5, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1 + c_dim, h), nn.SiLU(inplace=True),
            nn.Linear(h, h), nn.SiLU(inplace=True),
            ResBlock(h), ResBlock(h),
            nn.Linear(h, z_dim),
        )
    def forward(self, z, t, c): return self.net(torch.cat([z, t, c], dim=-1))

class CTIDiffNet(nn.Module):
    def __init__(self, z_dim=64, h=64):
        super().__init__()
        self.cti_gate = nn.Sequential(nn.Linear(1, h), nn.Softplus())
        self.state = nn.Sequential(nn.Linear(z_dim, h), nn.SiLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(h, z_dim), nn.Softplus())
    def forward(self, z, cti): return self.out(self.state(z) * self.cti_gate(cti))

class LatentNeuralSDE(nn.Module):
    def __init__(self, z_dim=64, c_dim=5, drift_h=256, diff_h=64, lambda_sigma=1.0):
        super().__init__()
        self.z_dim = z_dim; self.lambda_sigma = lambda_sigma
        self.drift = DriftNet(z_dim, c_dim, drift_h)
        self.diffusion = CTIDiffNet(z_dim, diff_h)
    def forward(self, z, t, c, cti):
        return self.drift(z, t, c), self.diffusion(z, cti)
    def sde_matching_loss(self, z, zn, t, c, cti, dt=1.0):
        mu = self.drift(z, t, c); sigma = self.diffusion(z, cti)
        dz = (zn - z) / dt
        drift_l = F.mse_loss(mu, dz)
        resid = (zn - z - mu * dt).pow(2) / dt
        diff_l = F.mse_loss(sigma.pow(2), resid)
        return {"loss": drift_l + self.lambda_sigma * diff_l,
                "drift": drift_l, "diffusion": diff_l}

# --- Score Decoder (NORMALIZED GHI, eps-parameterization) ---
GHI_SCALE = 1200.0

class ScoreRes(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(inplace=True), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)

class ScoreNet(nn.Module):
    def __init__(self, z_dim=64, c_dim=5, h=256, blocks=2):
        super().__init__()
        d_in = 1 + 1 + z_dim + 1 + c_dim
        layers = [nn.Linear(d_in, h), nn.SiLU(inplace=True)]
        for _ in range(blocks): layers.append(ScoreRes(h))
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, g, s, z, cti, c):
        return self.net(torch.cat([g, s, z, cti, c], dim=-1))

class CondScoreDecoder(nn.Module):
    def __init__(self, z_dim=64, c_dim=5, h=256, blocks=2, steps=100, b0=1e-4, b1=0.02):
        super().__init__()
        self.steps = steps
        self.score = ScoreNet(z_dim, c_dim, h, blocks)
        betas = torch.linspace(b0, b1, steps); alphas = 1 - betas
        ac = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cum", ac)
        self.register_buffer("sac", torch.sqrt(ac))
        self.register_buffer("s1mac", torch.sqrt(1 - ac))
    @staticmethod
    def _normalize(g_wm2): return g_wm2 / GHI_SCALE * 2.0 - 1.0
    @staticmethod
    def _denormalize(g_norm): return (g_norm + 1.0) / 2.0 * GHI_SCALE
    def training_loss(self, g0_wm2, z, cti, c):
        g0 = self._normalize(g0_wm2)
        B = g0.shape[0]
        si = torch.randint(0, self.steps, (B,), device=g0.device)
        sn = (si.float() / self.steps).unsqueeze(-1)
        eps = torch.randn_like(g0)
        gs = self.sac[si].unsqueeze(-1) * g0 + self.s1mac[si].unsqueeze(-1) * eps
        pred_noise = self.score(gs, sn, z, cti, c)
        return {"loss": F.mse_loss(pred_noise, eps)}
    @torch.no_grad()
    def sample(self, z, cti, c, n=1):
        B = z.shape[0]
        z_e = z.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
        cti_e = cti.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
        c_e = c.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
        x = torch.randn(B * n, 1, device=z.device)
        for i in reversed(range(self.steps)):
            sn = torch.full((B * n, 1), i / self.steps, device=z.device)
            eps_pred = self.score(x, sn, z_e, cti_e, c_e)
            b, a, ac = self.betas[i], self.alphas[i], self.alphas_cum[i]
            mean = (1 / a.sqrt()) * (x - b / (1 - ac).sqrt() * eps_pred)
            if i > 0: x = mean + b.sqrt() * torch.randn_like(x)
            else:     x = mean
        g_wm2 = self._denormalize(x).clamp(min=0.0, max=GHI_SCALE)
        return g_wm2.view(B, n)

# --- Metrics ---
def crps_empirical(y_true, y_samples):
    """y_true: (N,), y_samples: (N, M). Returns per-point CRPS (N,)."""
    N, M = y_samples.shape
    t1 = np.mean(np.abs(y_samples - y_true[:, None]), axis=1)
    ys = np.sort(y_samples, axis=1)
    w = 2 * np.arange(1, M + 1) - M - 1
    t2 = np.sum(w[None, :] * ys, axis=1) / (M * M)
    return t1 - t2

def picp_metric(y_true, y_samples, alpha=0.9):
    lo = np.quantile(y_samples, (1 - alpha) / 2, axis=1)
    hi = np.quantile(y_samples, 1 - (1 - alpha) / 2, axis=1)
    return float(((y_true >= lo) & (y_true <= hi)).mean())

def pinaw_metric(y_samples, y_range, alpha=0.9):
    lo = np.quantile(y_samples, (1 - alpha) / 2, axis=1)
    hi = np.quantile(y_samples, 1 - (1 - alpha) / 2, axis=1)
    return float((hi - lo).mean() / max(y_range, 1e-9))

def all_metrics(y_true, y_samples, is_ramp=None, alpha=0.9):
    if len(y_true) == 0: return {"crps": 0, "picp": 0, "pinaw": 0, "rmse": 0, "mae": 0, "ramp_crps": 0}
    y_med = np.median(y_samples, axis=1)
    y_range = float(y_true.max() - y_true.min())
    crps = crps_empirical(y_true, y_samples)
    out = {
        "crps":  float(crps.mean()),
        "picp":  picp_metric(y_true, y_samples, alpha),
        "pinaw": pinaw_metric(y_samples, y_range, alpha),
        "rmse":  float(np.sqrt(np.mean((y_true - y_med) ** 2))),
        "mae":   float(np.mean(np.abs(y_true - y_med))),
    }
    if is_ramp is not None and is_ramp.sum() > 0:
        out["ramp_crps"] = float(crps[is_ramp].mean())
    else:
        out["ramp_crps"] = 0.0
    return out

# --- SDE solver (with stability clamping) ---
_train_Z_np = np.load(LATENT_DIR / "train_latents.npy")
Z_MEAN = torch.from_numpy(_train_Z_np.mean(0)).float().to(DEVICE)
Z_STD_RAW = torch.from_numpy(_train_Z_np.std(0)).float().to(DEVICE) + 1e-6
Z_STD = torch.maximum(Z_STD_RAW, torch.full_like(Z_STD_RAW, 0.05))
Z_CLAMP_STDS = 8.0
MU_CAP = 10.0
SIGMA_CAP = 5.0
del _train_Z_np

def em_step(drift_fn, diff_fn, z, t, c, cti, dt):
    mu = drift_fn(z, t, c).clamp(-MU_CAP, MU_CAP)
    sigma = diff_fn(z, cti).clamp(0.0, SIGMA_CAP)
    z_new = z + mu * dt + sigma * (dt ** 0.5) * torch.randn_like(z)
    return torch.clamp(z_new, Z_MEAN - Z_CLAMP_STDS * Z_STD, Z_MEAN + Z_CLAMP_STDS * Z_STD)

def solve_sde_horizons(sde, z0, horizons, c, cti, N=50, dt=1.0):
    B, d = z0.shape
    mx = max(horizons); hset = set(horizons)
    z = z0.unsqueeze(1).expand(B, N, d).reshape(B * N, d)
    c_e = c.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
    cti_e = cti.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
    out = {}
    for step in range(mx):
        t = torch.full((B * N, 1), float(step), device=z0.device)
        z = em_step(sde.drift, sde.diffusion, z, t, c_e, cti_e, dt)
        if (step + 1) in hset: out[step + 1] = z.view(B, N, d).clone()
    return out

print("Shared code loaded.")
'''

LOAD_DATA_CODE = '''\
# ==== Load all data tensors ====
def load_split(s):
    return {
        "Z":    np.load(LATENT_DIR / f"{s}_latents.npy"),
        "cti":  np.load(LATENT_DIR / f"{s}_cti.npy"),
        "ghi":  np.load(LATENT_DIR / f"{s}_ghi.npy"),
        "cov":  np.load(LATENT_DIR / f"{s}_covariates.npy"),
        "ramp": np.load(LATENT_DIR / f"{s}_is_ramp.npy"),
    }
data = {s: load_split(s) for s in ["train", "val", "test"]}
for s, d in data.items():
    print(f"  {s}: Z={d['Z'].shape}, GHI=[{d['ghi'].min():.0f},{d['ghi'].max():.0f}], ramps={int(d['ramp'].sum())}")

train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
val_df   = pd.read_parquet(SPLITS_DIR / "val.parquet")
test_df  = pd.read_parquet(SPLITS_DIR / "test.parquet")
ext_train = pd.read_parquet(EXTENDED_DIR / "train.parquet")
ext_val   = pd.read_parquet(EXTENDED_DIR / "val.parquet")
print(f"\\n8-day image splits: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
print(f"90-day extended:    train={len(ext_train):,} val={len(ext_val):,}")

Z_DIM = data["train"]["Z"].shape[1]
C_DIM = max(1, data["train"]["cov"].shape[1])
print(f"\\nZ_DIM={Z_DIM}, C_DIM={C_DIM}")

HORIZONS = [6, 30, 60, 120, 180]
HORIZON_MIN = {6: 1, 30: 5, 60: 10, 120: 20, 180: 30}
N_SAMPLES = 50
N_EVAL = min(1000, len(data["test"]["Z"]) - max(HORIZONS) - 1)
SEQ_LEN = 30
print(f"Horizons: {list(HORIZON_MIN.values())} min, MC samples: {N_SAMPLES}, N_EVAL: {N_EVAL}")
'''

BASELINES_CODE = '''\
# ==== STAGE A: Baselines ====
STAGE_A_OUT = RESULTS_DIR / "main_results_combined.csv"
if STAGE_A_OUT.exists():
    print(f"[SKIP] Stage A already done: {STAGE_A_OUT}")
    combined = pd.read_csv(STAGE_A_OUT)
else:
    print("=" * 70)
    print("STAGE A: Training baselines")
    print("=" * 70)
    rng_global = np.random.default_rng(42); torch.manual_seed(42)
    all_baseline_results = {}

    # --- Load SolarSDE main results for the combined table ---
    if (RESULTS_DIR / "solar_sde_main_results.csv").exists():
        solar_df = pd.read_csv(RESULTS_DIR / "solar_sde_main_results.csv")
        solar_df["model"] = "SolarSDE"
    else:
        print("WARNING: solar_sde_main_results.csv missing — re-running main eval inline")
        # Fallback: run main eval here
        sde = LatentNeuralSDE(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
        sde.load_state_dict(torch.load(SDE_CKPT, map_location=DEVICE, weights_only=False)); sde.eval()
        score = CondScoreDecoder(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
        score.load_state_dict(torch.load(SCORE_CKPT, map_location=DEVICE, weights_only=False)); score.eval()
        te = data["test"]; res_s = {}
        for h in HORIZONS:
            yt, ys, rm = [], [], []
            for i in range(0, N_EVAL, 32):
                idx = list(range(i, min(i + 32, N_EVAL)))
                z0 = torch.from_numpy(te["Z"][idx]).float().to(DEVICE)
                c = torch.from_numpy(te["cov"][idx]).float().to(DEVICE)
                cti = torch.from_numpy(te["cti"][idx]).float().unsqueeze(-1).to(DEVICE)
                with torch.no_grad():
                    endp = solve_sde_horizons(sde, z0, [h], c, cti, N=N_SAMPLES)[h]
                    B, N, d = endp.shape
                    g = score.sample(endp.view(B*N, d),
                                     cti.unsqueeze(1).expand(B,N,-1).reshape(B*N,-1),
                                     c.unsqueeze(1).expand(B,N,-1).reshape(B*N,-1),
                                     n=1).squeeze(-1).view(B, N).cpu().numpy()
                for k, ii in enumerate(idx):
                    j = ii + h
                    if j < len(te["ghi"]):
                        yt.append(te["ghi"][j]); ys.append(g[k]); rm.append(te["ramp"][j])
            m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
            m["horizon_min"] = HORIZON_MIN[h]; m["horizon_steps"] = h; m["n_eval"] = len(yt)
            res_s[h] = m
        solar_df = pd.DataFrame.from_dict(res_s, orient="index").sort_values("horizon_min")
        solar_df.to_csv(RESULTS_DIR / "solar_sde_main_results.csv", index=False)
        solar_df["model"] = "SolarSDE"
        del sde, score; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def save_baseline(name, results_by_h):
        df = pd.DataFrame.from_dict(results_by_h, orient="index").sort_values("horizon_min")
        df["model"] = name
        df.to_csv(RESULTS_DIR / f"baseline_{name}_results.csv", index=False)
        all_baseline_results[name] = df

    te = data["test"]
    te_ghi = te["ghi"]; te_ramp = te["ramp"]

    # --- A1 Persistence ---
    print("\\n[A1] Persistence")
    tr_ghi = data["train"]["ghi"]
    pers_std = {h: float(np.std(tr_ghi[h:] - tr_ghi[:-h])) for h in HORIZONS}
    rng = np.random.default_rng(42)
    res_pers = {}
    for h in HORIZONS:
        yt, ys, rm = [], [], []
        for i in range(N_EVAL):
            if i + h < len(te_ghi):
                yp = te_ghi[i]
                samples = np.clip(yp + rng.normal(0, pers_std[h], size=N_SAMPLES), 0, None)
                yt.append(te_ghi[i + h]); ys.append(samples); rm.append(te_ramp[i + h])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["horizon_steps"] = h; m["n_eval"] = len(yt)
        res_pers[h] = m
        print(f"  h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} RMSE={m['rmse']:.2f} PICP={m['picp']:.3f}")
    save_baseline("persistence", res_pers)

    # --- A2 Smart Persistence ---
    print("\\n[A2] Smart Persistence")
    te_kt  = test_df["clear_sky_index"].values.astype(np.float32)
    te_gcs = test_df["ghi_clearsky"].values.astype(np.float32)
    tr_kt  = train_df["clear_sky_index"].values.astype(np.float32)
    tr_gcs = train_df["ghi_clearsky"].values.astype(np.float32)
    tr_ghi_df = train_df["ghi"].values.astype(np.float32)
    sp_std = {h: float(np.std(tr_ghi_df[h:] - tr_kt[:-h] * tr_gcs[h:])) for h in HORIZONS}
    rng = np.random.default_rng(42)
    res_sp = {}
    for h in HORIZONS:
        yt, ys, rm = [], [], []
        for i in range(N_EVAL):
            j = i + h
            if j < len(te_ghi) and j < len(te_gcs):
                pt = te_kt[i] * te_gcs[j]
                samples = np.clip(pt + rng.normal(0, sp_std[h], size=N_SAMPLES), 0, None)
                yt.append(te_ghi[j]); ys.append(samples); rm.append(te_ramp[j])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["horizon_steps"] = h; m["n_eval"] = len(yt)
        res_sp[h] = m
        print(f"  h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} RMSE={m['rmse']:.2f} PICP={m['picp']:.3f}")
    save_baseline("smart_persistence", res_sp)

    # --- Build LSTM sequence tensors from extended 90-day data ---
    print("\\n[A3/A4] Building LSTM sequence tensors (extended 90-day BMS)")
    def build_seq_tensors(df, seq_len, horizons):
        f_cols = ["ghi", "clear_sky_index", "solar_zenith"]
        for c in ["temperature", "humidity", "wind_speed"]:
            if c in df.columns: f_cols.append(c)
        X_arr = df[f_cols].fillna(0).values.astype(np.float32)
        ghi   = df["ghi"].values.astype(np.float32)
        mx = max(horizons)
        Xs, Ys = [], []
        for i in range(seq_len, len(X_arr) - mx):
            Xs.append(X_arr[i - seq_len:i])
            Ys.append(np.array([ghi[i + h] for h in horizons], dtype=np.float32))
        return torch.tensor(np.stack(Xs)), torch.tensor(np.stack(Ys))

    # Downsample extended 1-min BMS to 10s (keep every 6th row) for horizon alignment
    def ds(df): return df.iloc[::6].reset_index(drop=True) if len(df) > 0 else df
    Xtr, Ytr = build_seq_tensors(ds(ext_train), SEQ_LEN, HORIZONS)
    Xva, Yva = build_seq_tensors(ds(ext_val),   SEQ_LEN, HORIZONS)
    Xte, Yte = build_seq_tensors(test_df,       SEQ_LEN, HORIZONS)
    mu_f = Xtr.mean(dim=(0,1), keepdim=True); sd_f = Xtr.std(dim=(0,1), keepdim=True) + 1e-6
    Xtr_n = (Xtr - mu_f) / sd_f; Xva_n = (Xva - mu_f) / sd_f; Xte_n = (Xte - mu_f) / sd_f
    INPUT_DIM = Xtr_n.shape[-1]; N_H = len(HORIZONS)
    print(f"  Seq shapes: train={Xtr.shape}  val={Xva.shape}  test={Xte.shape}")
    te_ghi_seq = test_df["ghi"].values.astype(np.float32)
    te_ramp_seq = test_df["is_ramp"].values.astype(bool)

    class LSTMF(nn.Module):
        def __init__(self, d_in, h=128, nl=2, n_out=5, drop=0.0):
            super().__init__()
            self.lstm = nn.LSTM(d_in, h, nl, batch_first=True, dropout=drop if nl > 1 else 0.0)
            self.drop = nn.Dropout(drop); self.fc = nn.Linear(h, n_out)
        def forward(self, x):
            _, (hn, _) = self.lstm(x); return self.fc(self.drop(hn[-1]))

    def train_lstm(model, X, Y, Xv, Yv, epochs=40, bs=128, lr=1e-3, tag=""):
        model = model.to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr); crit = nn.MSELoss()
        dl = DataLoader(TensorDataset(X, Y), batch_size=bs, shuffle=True, drop_last=True)
        dv = DataLoader(TensorDataset(Xv, Yv), batch_size=bs)
        best = float("inf")
        for ep in range(1, epochs + 1):
            model.train(); tl = 0; n = 0
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = crit(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item(); n += 1
            model.eval(); vl = vn = 0
            with torch.no_grad():
                for xb, yb in dv:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    vl += crit(model(xb), yb).item(); vn += 1
            vl /= max(vn, 1)
            if vl < best:
                best = vl
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"{tag}_best.pt")
            if ep % 10 == 0 or ep == 1:
                print(f"    {tag} ep {ep}/{epochs} tr={tl/n:.4f} val={vl:.4f}")
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{tag}_best.pt", map_location=DEVICE, weights_only=False))
        return model

    # --- A3 LSTM deterministic ---
    print("\\n[A3] LSTM deterministic (40 epochs)")
    torch.manual_seed(42)
    lstm = train_lstm(LSTMF(INPUT_DIM, 128, 2, N_H, drop=0.0), Xtr_n, Ytr, Xva_n, Yva, epochs=40, tag="lstm_det")
    lstm.eval()
    with torch.no_grad():
        pred_tr = lstm(Xtr_n.to(DEVICE)).cpu().numpy()
        pred_te = lstm(Xte_n.to(DEVICE)).cpu().numpy()
    res_tr_lstm = Ytr.numpy() - pred_tr
    lstm_std = {HORIZONS[i]: float(res_tr_lstm[:, i].std()) for i in range(N_H)}
    rng = np.random.default_rng(42); res_lstm = {}
    for hi, h in enumerate(HORIZONS):
        yt, ys, rm = [], [], []
        for i in range(min(N_EVAL, len(pred_te))):
            ti = SEQ_LEN + i + h
            if ti < len(te_ghi_seq):
                pt = pred_te[i, hi]
                samples = np.clip(pt + rng.normal(0, lstm_std[h], size=N_SAMPLES), 0, None)
                yt.append(te_ghi_seq[ti]); ys.append(samples); rm.append(te_ramp_seq[ti])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["horizon_steps"] = h; m["n_eval"] = len(yt)
        res_lstm[h] = m
        print(f"  h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} RMSE={m['rmse']:.2f} PICP={m['picp']:.3f}")
    save_baseline("lstm", res_lstm)

    # --- A4 MC-Dropout LSTM ---
    print("\\n[A4] MC-Dropout LSTM (40 epochs)")
    torch.manual_seed(42)
    mcd = train_lstm(LSTMF(INPUT_DIM, 128, 2, N_H, drop=0.1), Xtr_n, Ytr, Xva_n, Yva, epochs=40, tag="lstm_mcd")
    def mc_predict(model, X, n_passes=50, bs=256):
        model.train()
        out = []
        for _ in range(n_passes):
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), bs):
                    preds.append(model(X[i:i+bs].to(DEVICE)).cpu())
            out.append(torch.cat(preds, dim=0).numpy())
        model.eval()
        return np.stack(out, axis=0)
    mc_pred = mc_predict(mcd, Xte_n, n_passes=N_SAMPLES)
    res_mcd = {}
    for hi, h in enumerate(HORIZONS):
        yt, ys, rm = [], [], []
        for i in range(min(N_EVAL, mc_pred.shape[1])):
            ti = SEQ_LEN + i + h
            if ti < len(te_ghi_seq):
                samples = np.clip(mc_pred[:, i, hi], 0, None)
                yt.append(te_ghi_seq[ti]); ys.append(samples); rm.append(te_ramp_seq[ti])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["horizon_steps"] = h; m["n_eval"] = len(yt)
        res_mcd[h] = m
        print(f"  h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} RMSE={m['rmse']:.2f} PICP={m['picp']:.3f}")
    save_baseline("mc_dropout", res_mcd)

    del lstm, mcd, pred_tr, pred_te, mc_pred
    gc.collect(); torch.cuda.is_available() and torch.cuda.empty_cache()

    # --- A5 CSDI (horizon-conditioned, trained once) ---
    print("\\n[A5] CSDI conditional diffusion (30 epochs, horizon-conditioned)")
    class DiffEmb(nn.Module):
        def __init__(self, d=64):
            super().__init__(); half = d // 2
            emb = math.log(10000) / (half - 1)
            self.register_buffer("emb", torch.exp(torch.arange(half).float() * -emb))
        def forward(self, t):
            e = t.unsqueeze(-1).float() * self.emb.unsqueeze(0)
            return torch.cat([e.sin(), e.cos()], dim=-1)
    class TxBlock(nn.Module):
        def __init__(self, d=64, nh=4):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
            self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        def forward(self, x):
            h = self.n1(x); h, _ = self.attn(h, h, h); x = x + h
            return x + self.ffn(self.n2(x))
    class CSDIScoreNet(nn.Module):
        """Horizon-conditioned CSDI with GHI normalization (same trick as SolarSDE's CSMID).
        Training targets are GHI/GHI_SCALE * 2 - 1 in [-1, 1]. Reverse sampling denormalizes.
        """
        def __init__(self, d_in, d=64, nh=4, nl=4, steps=100):
            super().__init__()
            self.steps = steps
            self.demb = DiffEmb(d); self.hemb = nn.Embedding(5, d)
            self.proj = nn.Linear(d_in + 1, d); self.dproj = nn.Linear(d, d)
            self.blocks = nn.ModuleList([TxBlock(d, nh) for _ in range(nl)])
            self.out = nn.Linear(d, 1)
            b = torch.linspace(1e-4, 0.02, steps); a = 1 - b; ac = torch.cumprod(a, 0)
            self.register_buffer("betas", b); self.register_buffer("alphas", a); self.register_buffer("ac", ac)
            self.register_buffer("sac", torch.sqrt(ac)); self.register_buffer("s1mac", torch.sqrt(1 - ac))
        @staticmethod
        def _norm(g_wm2): return g_wm2 / GHI_SCALE * 2.0 - 1.0   # uses GHI_SCALE=1200 from shared code
        @staticmethod
        def _denorm(g_norm): return (g_norm + 1.0) / 2.0 * GHI_SCALE
        def _forward(self, x_cond, y_noisy, t_idx, h_idx):
            B, S, D = x_cond.shape
            extra = torch.zeros(B, 1, D, device=x_cond.device); extra[:, 0, 0] = y_noisy.squeeze(-1)
            seq = torch.cat([x_cond, extra], dim=1)
            tgt = torch.zeros(B, S + 1, 1, device=x_cond.device); tgt[:, -1, 0] = y_noisy.squeeze(-1)
            h = self.proj(torch.cat([seq, tgt], dim=-1))
            te = self.demb(t_idx.float()); he = self.hemb(h_idx)
            h = h + self.dproj(te).unsqueeze(1) + he.unsqueeze(1)
            for blk in self.blocks: h = blk(h)
            return self.out(h[:, -1, :])
        def training_loss(self, x_cond, y_wm2, h_idx):
            """y_wm2 in W/m². Normalize to [-1, 1] before DSM."""
            y = self._norm(y_wm2)
            B = y.shape[0]; dev = y.device
            t = torch.randint(0, self.steps, (B,), device=dev)
            eps = torch.randn_like(y.unsqueeze(-1))
            yn = self.sac[t].unsqueeze(-1) * y.unsqueeze(-1) + self.s1mac[t].unsqueeze(-1) * eps
            pred = self._forward(x_cond, yn, t, h_idx)
            return F.mse_loss(pred, eps)
        @torch.no_grad()
        def sample(self, x_cond, h_idx, n=50):
            """Returns W/m² samples (denormalized + clamped)."""
            B = x_cond.shape[0]; dev = x_cond.device
            xc = x_cond.unsqueeze(1).expand(B, n, -1, -1).reshape(B * n, *x_cond.shape[1:])
            he = h_idx.unsqueeze(1).expand(B, n).reshape(B * n)
            x = torch.randn(B * n, 1, device=dev)
            for i in reversed(range(self.steps)):
                ti = torch.full((B * n,), i, device=dev, dtype=torch.long)
                eps_p = self._forward(xc, x, ti, he)
                b, a, ab = self.betas[i], self.alphas[i], self.ac[i]
                x = (1 / a.sqrt()) * (x - b / (1 - ab).sqrt() * eps_p)
                if i > 0: x = x + b.sqrt() * torch.randn_like(x)
            g_wm2 = self._denorm(x).clamp(0.0, GHI_SCALE)
            return g_wm2.squeeze(-1).view(B, n)

    torch.manual_seed(42)
    csdi = CSDIScoreNet(d_in=INPUT_DIM, d=64, nh=4, nl=4, steps=50).to(DEVICE)
    opt = torch.optim.Adam(csdi.parameters(), lr=1e-3)
    # Build multi-horizon training set: stack (X, Y[:, hi], hi) for each horizon
    multi_X = []; multi_Y = []; multi_H = []
    for hi in range(N_H):
        multi_X.append(Xtr_n); multi_Y.append(Ytr[:, hi]); multi_H.append(torch.full((len(Xtr_n),), hi, dtype=torch.long))
    multi_X = torch.cat(multi_X, 0); multi_Y = torch.cat(multi_Y, 0); multi_H = torch.cat(multi_H, 0)
    ds = TensorDataset(multi_X, multi_Y, multi_H)
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True, num_workers=0)
    EPOCHS_CSDI = 30
    t0 = time.time()
    for ep in range(1, EPOCHS_CSDI + 1):
        csdi.train(); tl = 0; n = 0
        for xb, yb, hb in dl:
            xb, yb, hb = xb.to(DEVICE), yb.to(DEVICE), hb.to(DEVICE)
            l = csdi.training_loss(xb, yb, hb)
            opt.zero_grad(); l.backward(); opt.step()
            tl += l.item(); n += 1
        if ep % 5 == 0 or ep == 1:
            print(f"    CSDI ep {ep}/{EPOCHS_CSDI}  loss={tl/n:.4f}  time={(time.time()-t0)/60:.1f}min")
    torch.save(csdi.state_dict(), CHECKPOINT_DIR / "csdi_best.pt")

    csdi.eval()
    res_csdi = {}
    for hi, h in enumerate(HORIZONS):
        print(f"  CSDI eval h={HORIZON_MIN[h]}min ...")
        yt, ys, rm = [], [], []
        bs = 4
        for i in range(0, min(N_EVAL, len(Xte_n)), bs):
            xb = Xte_n[i:i + bs].to(DEVICE)
            hb = torch.full((len(xb),), hi, dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                samp = csdi.sample(xb, hb, n=N_SAMPLES).cpu().numpy()
            for k in range(samp.shape[0]):
                ti = SEQ_LEN + i + k + h
                if ti < len(te_ghi_seq):
                    yt.append(te_ghi_seq[ti])
                    ys.append(samp[k])      # already in W/m², clamped to [0, GHI_SCALE]
                    rm.append(te_ramp_seq[ti])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["horizon_steps"] = h; m["n_eval"] = len(yt)
        res_csdi[h] = m
        print(f"    h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} RMSE={m['rmse']:.2f} PICP={m['picp']:.3f}")
    save_baseline("csdi", res_csdi)

    del csdi, ds, dl; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Combine ---
    parts = [solar_df]
    for name in ["persistence", "smart_persistence", "lstm", "mc_dropout", "csdi"]:
        parts.append(all_baseline_results[name])
    combined = pd.concat(parts, ignore_index=True)
    cols_keep = ["model", "horizon_min", "crps", "rmse", "mae", "picp", "pinaw", "ramp_crps"]
    combined = combined[[c for c in cols_keep if c in combined.columns]]
    combined = combined.sort_values(["model", "horizon_min"]).reset_index(drop=True)
    pers = combined[combined["model"] == "persistence"].set_index("horizon_min")["crps"].to_dict()
    combined["skill_vs_persistence"] = combined.apply(
        lambda r: 1 - r["crps"] / pers[r["horizon_min"]], axis=1
    )
    combined.to_csv(STAGE_A_OUT, index=False)

    print("\\n" + "=" * 80)
    print("STAGE A COMPLETE — main results table")
    print("=" * 80)
    print(combined.to_string(index=False))
'''

ABLATIONS_CODE = '''\
# ==== STAGE B: Ablations ====
STAGE_B_OUT = RESULTS_DIR / "ablation_results.csv"
if STAGE_B_OUT.exists():
    print(f"[SKIP] Stage B already done: {STAGE_B_OUT}")
    abl = pd.read_csv(STAGE_B_OUT)
else:
    print("=" * 70)
    print("STAGE B: Ablations (A2 no-CTI, A4 no-score, A5 no-SDE)")
    print("=" * 70)

    class LatentSeqDataset(Dataset):
        def __init__(self, d):
            self.Z=d["Z"]; self.cti=d["cti"]; self.ghi=d["ghi"]; self.cov=d["cov"]
        def __len__(self): return max(0, len(self.Z) - 1)
        def __getitem__(self, i):
            return {"z_t": torch.from_numpy(self.Z[i]).float(),
                    "z_next": torch.from_numpy(self.Z[i+1]).float(),
                    "cti": torch.tensor(float(self.cti[i])),
                    "ghi": torch.tensor(float(self.ghi[i])),
                    "cov": torch.from_numpy(self.cov[i]).float() if self.cov.shape[1] > 0
                           else torch.zeros(C_DIM)}
    tr_ds = LatentSeqDataset(data["train"]); va_ds = LatentSeqDataset(data["val"])

    # Keep reference to original SDE & Score for ablations that reuse them
    sde_full = LatentNeuralSDE(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
    sde_full.load_state_dict(torch.load(SDE_CKPT, map_location=DEVICE, weights_only=False)); sde_full.eval()
    score_full = CondScoreDecoder(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
    score_full.load_state_dict(torch.load(SCORE_CKPT, map_location=DEVICE, weights_only=False)); score_full.eval()

    # --- A2: no-CTI — retrain SDE with constant CTI=0 ---
    print("\\n[B-A2] Retraining SDE with CTI=0 ...")
    A2_SDE = CHECKPOINT_DIR / "sde_a2_best.pt"
    if A2_SDE.exists():
        print(f"  [skip retrain, using {A2_SDE}]")
        sde_a2 = LatentNeuralSDE(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
        sde_a2.load_state_dict(torch.load(A2_SDE, map_location=DEVICE, weights_only=False)); sde_a2.eval()
    else:
        torch.manual_seed(42); np.random.seed(42)
        sde_a2 = LatentNeuralSDE(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
        opt = torch.optim.Adam(sde_a2.parameters(), lr=1e-4)
        dl = DataLoader(tr_ds, batch_size=128, shuffle=True, drop_last=True)
        vl = DataLoader(va_ds, batch_size=128, shuffle=False)
        best = float("inf"); t0 = time.time()
        for ep in range(1, 101):
            sde_a2.train(); tl = 0; n = 0
            for b in dl:
                z = b["z_t"].to(DEVICE); zn = b["z_next"].to(DEVICE)
                cti0 = torch.zeros(z.shape[0], 1, device=DEVICE)
                c = b["cov"].to(DEVICE); t = torch.zeros(z.shape[0], 1, device=DEVICE)
                l = sde_a2.sde_matching_loss(z, zn, t, c, cti0)["loss"]
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(sde_a2.parameters(), 1.0); opt.step()
                tl += l.item(); n += 1
            sde_a2.eval(); vl_s = vn = 0
            with torch.no_grad():
                for b in vl:
                    z = b["z_t"].to(DEVICE); zn = b["z_next"].to(DEVICE)
                    cti0 = torch.zeros(z.shape[0], 1, device=DEVICE)
                    c = b["cov"].to(DEVICE); t = torch.zeros(z.shape[0], 1, device=DEVICE)
                    vl_s += sde_a2.sde_matching_loss(z, zn, t, c, cti0)["loss"].item(); vn += 1
            vl_s /= max(vn, 1)
            if vl_s < best: best = vl_s; torch.save(sde_a2.state_dict(), A2_SDE)
            if ep % 20 == 0: print(f"    A2 ep {ep}: train={tl/n:.5f} val={vl_s:.5f} t={(time.time()-t0)/60:.1f}m")
        sde_a2.load_state_dict(torch.load(A2_SDE, map_location=DEVICE, weights_only=False)); sde_a2.eval()

    te = data["test"]; res_a2 = {}
    for h in HORIZONS:
        yt, ys, rm = [], [], []
        for i in range(0, N_EVAL, 32):
            idx = list(range(i, min(i + 32, N_EVAL)))
            z0 = torch.from_numpy(te["Z"][idx]).float().to(DEVICE)
            c = torch.from_numpy(te["cov"][idx]).float().to(DEVICE)
            cti0 = torch.zeros(len(idx), 1, device=DEVICE)
            with torch.no_grad():
                endp = solve_sde_horizons(sde_a2, z0, [h], c, cti0, N=N_SAMPLES)[h]
                B, N, d = endp.shape
                g = score_full.sample(
                    endp.view(B*N, d),
                    cti0.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1),
                    c.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1), n=1
                ).squeeze(-1).view(B, N).cpu().numpy()
            for k, ii in enumerate(idx):
                j = ii + h
                if j < len(te["ghi"]):
                    yt.append(te["ghi"][j]); ys.append(g[k]); rm.append(te["ramp"][j])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["variant"] = "A2_no_cti"
        res_a2[h] = m
        print(f"  A2 h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} PICP={m['picp']:.3f}")
    pd.DataFrame.from_dict(res_a2, orient="index").sort_values("horizon_min").to_csv(
        RESULTS_DIR / "ablation_a2_no_cti.csv", index=False)

    # --- A4: linear decoder z->GHI, no score matching ---
    print("\\n[B-A4] Training linear decoder ...")
    A4_LIN = CHECKPOINT_DIR / "linear_decoder_a4.pt"
    class LinearDecoder(nn.Module):
        def __init__(self, z_dim, c_dim, h=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(z_dim + 1 + c_dim, h), nn.SiLU(), nn.Linear(h, 1))
        def forward(self, z, cti, c): return self.net(torch.cat([z, cti, c], dim=-1)).squeeze(-1)
    torch.manual_seed(42)
    lin = LinearDecoder(Z_DIM, C_DIM, 64).to(DEVICE)
    if A4_LIN.exists():
        print(f"  [skip retrain]"); lin.load_state_dict(torch.load(A4_LIN, map_location=DEVICE, weights_only=False))
    else:
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3); crit = nn.MSELoss()
        dl = DataLoader(tr_ds, batch_size=256, shuffle=True, drop_last=True)
        for ep in range(1, 41):
            lin.train(); tl = 0; n = 0
            for b in dl:
                z = b["z_t"].to(DEVICE); cti = b["cti"].to(DEVICE).unsqueeze(-1)
                c = b["cov"].to(DEVICE); g = b["ghi"].to(DEVICE)
                loss = crit(lin(z, cti, c), g)
                opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); n += 1
            if ep % 10 == 0: print(f"    A4 ep {ep}: tr={tl/n:.3f}")
        torch.save(lin.state_dict(), A4_LIN)
    lin.eval()

    # Residual std calibrated on VAL (not train) to avoid leakage
    with torch.no_grad():
        z_va = torch.from_numpy(data["val"]["Z"]).float().to(DEVICE)
        cti_va = torch.from_numpy(data["val"]["cti"]).float().unsqueeze(-1).to(DEVICE)
        c_va = torch.from_numpy(data["val"]["cov"]).float().to(DEVICE)
        pred_va = lin(z_va, cti_va, c_va).cpu().numpy()
    a4_std = float(np.std(data["val"]["ghi"] - pred_va))
    print(f"  A4 val residual std: {a4_std:.2f} W/m²")

    rng = np.random.default_rng(42); res_a4 = {}
    for h in HORIZONS:
        yt, ys, rm = [], [], []
        for i in range(0, N_EVAL, 32):
            idx = list(range(i, min(i + 32, N_EVAL)))
            z0 = torch.from_numpy(te["Z"][idx]).float().to(DEVICE)
            c = torch.from_numpy(te["cov"][idx]).float().to(DEVICE)
            cti = torch.from_numpy(te["cti"][idx]).float().unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                endp = solve_sde_horizons(sde_full, z0, [h], c, cti, N=N_SAMPLES)[h]
                B, N, d = endp.shape
                cti_e = cti.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
                c_e = c.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
                pred = lin(endp.view(B * N, d), cti_e, c_e).view(B, N).cpu().numpy()
                noise = rng.normal(0, a4_std, size=(B, N))
                g = np.clip(pred + noise, 0, None)
            for k, ii in enumerate(idx):
                j = ii + h
                if j < len(te["ghi"]):
                    yt.append(te["ghi"][j]); ys.append(g[k]); rm.append(te["ramp"][j])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["variant"] = "A4_no_score"
        res_a4[h] = m
        print(f"  A4 h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} PICP={m['picp']:.3f}")
    pd.DataFrame.from_dict(res_a4, orient="index").sort_values("horizon_min").to_csv(
        RESULTS_DIR / "ablation_a4_no_score.csv", index=False)

    # --- A5: no SDE (deterministic ODE, drift only) ---
    print("\\n[B-A5] Training deterministic drift-only ODE ...")
    A5_CKPT = CHECKPOINT_DIR / "sde_a5_best.pt"
    torch.manual_seed(42)
    sde_a5 = LatentNeuralSDE(z_dim=Z_DIM, c_dim=C_DIM, lambda_sigma=0.0).to(DEVICE)
    if A5_CKPT.exists():
        print("  [skip retrain]"); sde_a5.load_state_dict(torch.load(A5_CKPT, map_location=DEVICE, weights_only=False))
    else:
        opt = torch.optim.Adam(sde_a5.drift.parameters(), lr=1e-4)
        dl = DataLoader(tr_ds, batch_size=128, shuffle=True, drop_last=True)
        for ep in range(1, 101):
            sde_a5.train(); tl = 0; n = 0
            for b in dl:
                z = b["z_t"].to(DEVICE); zn = b["z_next"].to(DEVICE)
                c = b["cov"].to(DEVICE); t = torch.zeros(z.shape[0], 1, device=DEVICE)
                mu = sde_a5.drift(z, t, c); l = F.mse_loss(mu, (zn - z) / 1.0)
                opt.zero_grad(); l.backward(); opt.step(); tl += l.item(); n += 1
            if ep % 20 == 0: print(f"    A5 ep {ep}: drift_loss={tl/n:.5f}")
        torch.save(sde_a5.state_dict(), A5_CKPT)
    sde_a5.eval()

    def solve_ode_horizons(drift_fn, z0, horizons, c, dt=1.0):
        B, d = z0.shape; mx = max(horizons); hset = set(horizons); out = {}
        z = z0.clone()
        for step in range(mx):
            t = torch.full((B, 1), float(step), device=z.device)
            z = torch.clamp(z + drift_fn(z, t, c) * dt,
                            Z_MEAN - Z_CLAMP_STDS * Z_STD, Z_MEAN + Z_CLAMP_STDS * Z_STD)
            if (step + 1) in hset: out[step + 1] = z.clone()
        return out

    res_a5 = {}
    for h in HORIZONS:
        yt, ys, rm = [], [], []
        for i in range(0, N_EVAL, 32):
            idx = list(range(i, min(i + 32, N_EVAL)))
            z0 = torch.from_numpy(te["Z"][idx]).float().to(DEVICE)
            c = torch.from_numpy(te["cov"][idx]).float().to(DEVICE)
            cti = torch.from_numpy(te["cti"][idx]).float().unsqueeze(-1).to(DEVICE)
            with torch.no_grad():
                endp = solve_ode_horizons(sde_a5.drift, z0, [h], c, dt=1.0)[h]
                B = len(idx)
                endp_rep = endp.unsqueeze(1).expand(-1, N_SAMPLES, -1).reshape(-1, Z_DIM)
                cti_rep = cti.unsqueeze(1).expand(-1, N_SAMPLES, -1).reshape(-1, 1)
                c_rep = c.unsqueeze(1).expand(-1, N_SAMPLES, -1).reshape(-1, C_DIM)
                g = score_full.sample(endp_rep, cti_rep, c_rep, n=1).squeeze(-1).view(B, N_SAMPLES).cpu().numpy()
            for k, ii in enumerate(idx):
                j = ii + h
                if j < len(te["ghi"]):
                    yt.append(te["ghi"][j]); ys.append(g[k]); rm.append(te["ramp"][j])
        m = all_metrics(np.array(yt), np.array(ys), is_ramp=np.array(rm))
        m["horizon_min"] = HORIZON_MIN[h]; m["variant"] = "A5_deterministic_ode"
        res_a5[h] = m
        print(f"  A5 h={HORIZON_MIN[h]}min: CRPS={m['crps']:.2f} PICP={m['picp']:.3f}")
    pd.DataFrame.from_dict(res_a5, orient="index").sort_values("horizon_min").to_csv(
        RESULTS_DIR / "ablation_a5_det_ode.csv", index=False)

    # Combine ablations + A1 (full model) into one table
    a1 = pd.read_csv(RESULTS_DIR / "solar_sde_main_results.csv").copy()
    a1["variant"] = "A1_full"
    df_a2 = pd.read_csv(RESULTS_DIR / "ablation_a2_no_cti.csv")
    df_a4 = pd.read_csv(RESULTS_DIR / "ablation_a4_no_score.csv")
    df_a5 = pd.read_csv(RESULTS_DIR / "ablation_a5_det_ode.csv")
    abl = pd.concat([a1, df_a2, df_a4, df_a5], ignore_index=True)
    cols = ["variant", "horizon_min", "crps", "rmse", "mae", "picp", "pinaw", "ramp_crps"]
    abl = abl[[c for c in cols if c in abl.columns]]
    abl.to_csv(STAGE_B_OUT, index=False)

    del sde_a2, sde_a5, lin; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\\n" + "=" * 70); print("STAGE B COMPLETE")
    print(abl.to_string(index=False))
'''

CALIBRATION_CODE = '''\
# ==== STAGE C: Post-hoc Conformal Calibration + Analysis ====
STAGE_C_OUT = RESULTS_DIR / "solar_sde_calibrated.csv"
if STAGE_C_OUT.exists():
    print(f"[SKIP] Stage C already done: {STAGE_C_OUT}")
    df_cal = pd.read_csv(STAGE_C_OUT)
else:
    print("=" * 70)
    print("STAGE C: Conformal calibration + per-point predictions for analysis")
    print("=" * 70)

    sde = LatentNeuralSDE(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
    sde.load_state_dict(torch.load(SDE_CKPT, map_location=DEVICE, weights_only=False)); sde.eval()
    score = CondScoreDecoder(z_dim=Z_DIM, c_dim=C_DIM).to(DEVICE)
    score.load_state_dict(torch.load(SCORE_CKPT, map_location=DEVICE, weights_only=False)); score.eval()

    def gen_forecasts(split_data, n_eval, horizons):
        res = {}
        for h in horizons:
            yt, ys, rm = [], [], []
            for i in range(0, n_eval, 32):
                idx = list(range(i, min(i + 32, n_eval)))
                z0 = torch.from_numpy(split_data["Z"][idx]).float().to(DEVICE)
                c = torch.from_numpy(split_data["cov"][idx]).float().to(DEVICE)
                cti = torch.from_numpy(split_data["cti"][idx]).float().unsqueeze(-1).to(DEVICE)
                with torch.no_grad():
                    endp = solve_sde_horizons(sde, z0, [h], c, cti, N=N_SAMPLES)[h]
                    B, N, d = endp.shape
                    g = score.sample(endp.view(B*N, d),
                                     cti.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1),
                                     c.unsqueeze(1).expand(B, N, -1).reshape(B*N, -1),
                                     n=1).squeeze(-1).view(B, N).cpu().numpy()
                for k, ii in enumerate(idx):
                    j = ii + h
                    if j < len(split_data["ghi"]):
                        yt.append(split_data["ghi"][j]); ys.append(g[k]); rm.append(split_data["ramp"][j])
            res[h] = {"yt": np.array(yt), "ys": np.array(ys), "ramp": np.array(rm)}
        return res

    N_VAL_CAL = min(500, len(data["val"]["Z"]) - max(HORIZONS) - 1)
    print(f"Generating val forecasts ({N_VAL_CAL} points) for calibration ...")
    val_f = gen_forecasts(data["val"], N_VAL_CAL, HORIZONS)

    # Split-conformal quantile of |y - median|
    ALPHA = 0.10
    conformal_q = {}
    for h in HORIZONS:
        fv = val_f[h]
        med = np.median(fv["ys"], axis=1)
        r = np.abs(fv["yt"] - med)
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - ALPHA)))
        q = float(np.sort(r)[min(k - 1, n - 1)]) if n > 0 else 0.0
        conformal_q[h] = q
        print(f"  h={HORIZON_MIN[h]}: conformal q_90% = {q:.2f} W/m²")

    print(f"\\nGenerating test forecasts ({N_EVAL} points) ...")
    test_f = gen_forecasts(data["test"], N_EVAL, HORIZONS)

    cal_rows = []
    for h in HORIZONS:
        tf = test_f[h]
        yt, ys = tf["yt"], tf["ys"]
        med = np.median(ys, axis=1)
        raw = all_metrics(yt, ys, is_ramp=tf["ramp"])

        q = conformal_q[h]
        lo = med - q; hi = med + q
        cal_picp = float(((yt >= lo) & (yt <= hi)).mean())
        yrange = float(yt.max() - yt.min()); cal_pinaw = float((hi - lo).mean() / max(yrange, 1e-9))

        # Variance-inflated CRPS: rescale samples around median so std matches half-width
        raw_sd = ys.std(axis=1)
        target_sd = q / 1.645
        scale = np.where(raw_sd > 1e-3, target_sd / np.maximum(raw_sd, 1e-3), 1.0)
        ys_cal = med[:, None] + (ys - med[:, None]) * scale[:, None]
        cal_crps = float(crps_empirical(yt, ys_cal).mean())

        cal_rows.append({
            "horizon_min": HORIZON_MIN[h],
            "raw_crps": raw["crps"], "cal_crps": cal_crps,
            "raw_picp": raw["picp"], "cal_picp": cal_picp,
            "raw_pinaw": raw["pinaw"], "cal_pinaw": cal_pinaw,
            "rmse": raw["rmse"], "mae": raw["mae"], "ramp_crps": raw["ramp_crps"],
            "conformal_q_Wm2": q,
        })

    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_csv(STAGE_C_OUT, index=False)
    print("\\nCalibrated results:")
    print(df_cal.to_string(index=False))

    # Save per-point predictions for 10-min horizon (used in analysis)
    H_ANALYSIS = 60
    tf = test_f[H_ANALYSIS]
    np.savez(RESULTS_DIR / "test_predictions_h10min.npz",
             y_true=tf["yt"], y_samples=tf["ys"], is_ramp=tf["ramp"])
    print(f"\\nSaved per-point predictions to test_predictions_h10min.npz")

    del sde, score; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
'''

ANALYSIS_CODE = '''\
# ==== STAGE D: CTI analysis + economic value + figures ====
STAGE_D_OUT = FIGURES_DIR / "fig2_crps_vs_horizon.pdf"
if STAGE_D_OUT.exists():
    print(f"[SKIP] Stage D done (figures exist).")
else:
    print("=" * 70)
    print("STAGE D: Analysis + figures")
    print("=" * 70)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as spstats
    from sklearn.cluster import KMeans

    test_predictions = np.load(RESULTS_DIR / "test_predictions_h10min.npz")
    yt = test_predictions["y_true"]; ys = test_predictions["y_samples"]; is_ramp = test_predictions["is_ramp"]
    crps_per = crps_empirical(yt, ys)
    med = np.median(ys, axis=1)

    # CTI analysis
    print("\\n[D1] CTI analysis")
    cti_test = data["test"]["cti"]; ghi_test = data["test"]["ghi"]
    window = 6
    ghi_std = np.zeros_like(ghi_test)
    for t in range(window, len(ghi_test)):
        ghi_std[t] = np.std(ghi_test[t - window:t])
    mask = (cti_test > 0) & (ghi_std > 0)
    rho, pv = spstats.spearmanr(cti_test[mask], ghi_std[mask])
    print(f"  CTI vs GHI-var Spearman rho={rho:.3f}, p={pv:.2e}, N={int(mask.sum())}")

    # Align CTI to eval indices (test evaluation indices map to test_Z[0..N_EVAL])
    cti_eval = cti_test[:len(yt)]
    valid = cti_eval > 0
    if valid.sum() > 4:
        qs = np.quantile(cti_eval[valid], np.linspace(0, 1, 5))
        quartile_stats = []
        for i in range(4):
            m = (cti_eval >= qs[i]) & (cti_eval < qs[i + 1] if i < 3 else cti_eval <= qs[i + 1])
            if m.sum() > 0:
                quartile_stats.append({"quartile": i + 1,
                                      "cti_mean": float(cti_eval[m].mean()),
                                      "crps_mean": float(crps_per[m].mean()),
                                      "n": int(m.sum())})
    else:
        quartile_stats = []
    for s in quartile_stats:
        print(f"  Q{s['quartile']}: CTI={s['cti_mean']:.4f}, CRPS={s['crps_mean']:.2f}, N={s['n']}")

    # K-means regimes
    valid_cti = cti_test[cti_test > 0].reshape(-1, 1)
    if len(valid_cti) > 4:
        km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(valid_cti)
        centers_sorted = np.argsort(km.cluster_centers_.flatten())
        regime_names = ["Clear", "Thin Cloud", "Broken Cloud", "Overcast"]
        regime_stats = []
        ghi_valid = ghi_test[cti_test > 0]
        for i, name in enumerate(regime_names):
            ci = centers_sorted[i]
            mm = km.labels_ == ci
            regime_stats.append({"regime": name,
                                "cti_center": float(km.cluster_centers_.flatten()[ci]),
                                "n": int(mm.sum()),
                                "ghi_mean": float(ghi_valid[mm].mean()),
                                "ghi_std": float(ghi_valid[mm].std())})
        for s in regime_stats:
            print(f"  {s['regime']}: CTI={s['cti_center']:.4f}, GHI={s['ghi_mean']:.1f}±{s['ghi_std']:.1f}, N={s['n']}")
    else:
        regime_stats = []

    (RESULTS_DIR / "cti_analysis.json").write_text(json.dumps({
        "spearman_rho": float(rho), "spearman_p": float(pv),
        "quartile_stats": quartile_stats, "regime_stats": regime_stats,
    }, indent=2))

    # --- Economic value simulation ---
    print("\\n[D2] Economic value")
    def simulate_cost(y_true, y_samples, q=0.95, rcost=50.0, pcost=1000.0,
                      dec_min=5, plant_mw=1000.0, dt_s=10):
        steps_per = (dec_min * 60) // dt_s
        reserve = np.quantile(y_samples, q, axis=1)
        idx = np.arange(0, len(y_true), steps_per)
        rc = pc = 0.0
        for i in idx:
            res_mw = (reserve[i] / 1000.0) * plant_mw
            act_mw = (y_true[i] / 1000.0) * plant_mw
            hrs = dec_min / 60
            rc += res_mw * rcost * hrs
            if act_mw > res_mw: pc += (act_mw - res_mw) * pcost * hrs
        tot = rc + pc
        test_h = len(y_true) * dt_s / 3600
        ann = 365.25 * 12 / max(test_h, 1e-3)
        return {"reserve": float(rc), "penalty": float(pc), "total": float(tot),
                "annual_total": float(tot * ann),
                "annual_per_gw": float(tot * ann / (plant_mw / 1000))}

    cost_solar = simulate_cost(yt, ys)
    # Persistence baseline for comparison (use same test points)
    h_steps = 60
    tr_ghi = data["train"]["ghi"]
    pers_std = float(np.std(tr_ghi[h_steps:] - tr_ghi[:-h_steps]))
    rng = np.random.default_rng(42)
    ys_pers = np.zeros_like(ys)
    for i in range(len(yt)):
        gc_ = data["test"]["ghi"][i] if i < len(data["test"]["ghi"]) else yt[i]
        ys_pers[i] = np.clip(gc_ + rng.normal(0, pers_std, size=N_SAMPLES), 0, None)
    cost_pers = simulate_cost(yt, ys_pers)
    savings = {
        "annual_per_gw": cost_pers["annual_per_gw"] - cost_solar["annual_per_gw"],
        "pct": (cost_pers["total"] - cost_solar["total"]) / max(cost_pers["total"], 1) * 100,
    }
    print(f"  SolarSDE annual: ${cost_solar['annual_per_gw']/1e6:.2f}M/GW")
    print(f"  Persistence:     ${cost_pers['annual_per_gw']/1e6:.2f}M/GW")
    print(f"  Savings:         ${savings['annual_per_gw']/1e6:.2f}M/GW/yr  ({savings['pct']:.1f}% reduction)")
    (RESULTS_DIR / "economic_value.json").write_text(json.dumps({
        "solar_sde": cost_solar, "persistence": cost_pers, "savings": savings}, indent=2))

    # --- Reliability diagram data + PIT ---
    pit = np.mean(ys <= yt[:, None], axis=1)
    levels = np.arange(0.1, 1.0, 0.1)
    observed = []
    for L in levels:
        lo = np.quantile(ys, (1 - L) / 2, axis=1); hi = np.quantile(ys, 1 - (1 - L) / 2, axis=1)
        observed.append(float(((yt >= lo) & (yt <= hi)).mean()))
    (RESULTS_DIR / "reliability_data.json").write_text(json.dumps({
        "nominal": levels.tolist(), "observed": observed}, indent=2))

    # ===== FIGURES =====
    print("\\n[D3] Generating figures")

    # Fig 2: CRPS vs horizon
    combined = pd.read_csv(RESULTS_DIR / "main_results_combined.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {"SolarSDE": ("#e74c3c", 2.5), "persistence": ("#95a5a6", 1.0),
             "smart_persistence": ("#7f8c8d", 1.0), "lstm": ("#3498db", 1.5),
             "mc_dropout": ("#2980b9", 1.5), "csdi": ("#9b59b6", 1.5)}
    for m, (col, lw) in style.items():
        sub = combined[combined["model"] == m].sort_values("horizon_min")
        if len(sub) > 0:
            ax.plot(sub["horizon_min"], sub["crps"], "o-", color=col, linewidth=lw, label=m)
    ax.set_xlabel("Forecast Horizon (min)"); ax.set_ylabel("CRPS (W/m²)")
    ax.set_title("Probabilistic Forecast Performance"); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig2_crps_vs_horizon.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved fig2_crps_vs_horizon.pdf")

    # Fig 3a: CTI scatter
    if mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        idx_plot = np.random.choice(np.where(mask)[0], min(3000, int(mask.sum())), replace=False)
        ax.scatter(cti_test[idx_plot], ghi_std[idx_plot], alpha=0.3, s=5, c="#3498db")
        ax.set_xlabel("CTI"); ax.set_ylabel("GHI rolling std (W/m²)")
        ax.set_title(f"CTI vs Irradiance Variability (ρ={rho:.3f})"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig3a_cti_scatter.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig); print("  saved fig3a_cti_scatter.pdf")

    # Fig 3b: CRPS by CTI quartile
    if quartile_stats:
        fig, ax = plt.subplots(figsize=(6, 4))
        lbls = [f"Q{s['quartile']}\\n(CTI={s['cti_mean']:.3f})" for s in quartile_stats]
        vals = [s["crps_mean"] for s in quartile_stats]
        cols = plt.cm.YlOrRd(np.linspace(0.3, 0.85, len(lbls)))
        ax.bar(lbls, vals, color=cols, edgecolor="white")
        ax.set_ylabel("Mean CRPS (W/m²)"); ax.set_title("CRPS by CTI Quartile")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig3b_crps_by_cti.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig); print("  saved fig3b_crps_by_cti.pdf")

    # Fig 5: Reliability diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
    ax.plot(levels, observed, "o-", color="#e74c3c", linewidth=2.5, label="SolarSDE (raw)")
    # Also show calibrated line assuming perfect PICP at 90% after conformal
    ax.set_xlabel("Nominal Coverage"); ax.set_ylabel("Observed Coverage")
    ax.set_title("Reliability Diagram"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig5_reliability.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved fig5_reliability.pdf")

    # Fig 6: Economic value
    fig, ax = plt.subplots(figsize=(7, 4))
    mm = ["Persistence", "SolarSDE"]; costs = [cost_pers["annual_per_gw"]/1e6, cost_solar["annual_per_gw"]/1e6]
    ax.bar(mm, costs, color=["#95a5a6", "#e74c3c"], edgecolor="white")
    ax.set_ylabel("Annual Reserve Cost ($M / GW)")
    ax.set_title(f"Economic Value (Savings: ${savings['annual_per_gw']/1e6:.2f}M/GW/yr)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig6_economic_value.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved fig6_economic_value.pdf")

    # PIT histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit, bins=10, range=(0, 1), density=True, color="#3498db", edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT"); ax.set_ylabel("Density"); ax.set_title("PIT Histogram (SolarSDE)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig_pit_histogram.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  saved fig_pit_histogram.pdf")

    print(f"\\nAll figures saved to: {FIGURES_DIR}")

# ==== Final paper tables ====
try:
    combined = pd.read_csv(RESULTS_DIR / "main_results_combined.csv")
    t1 = combined[combined["horizon_min"] == 10]
    t1.to_csv(RESULTS_DIR / "paper_table1_main.csv", index=False)
    print("\\n=== PAPER TABLE 1 — main results at h=10min ===")
    print(t1.to_string(index=False))
except Exception as e:
    print(f"Table 1 error: {e}")

try:
    abl = pd.read_csv(RESULTS_DIR / "ablation_results.csv")
    t2 = abl[abl["horizon_min"] == 10]
    t2.to_csv(RESULTS_DIR / "paper_table2_ablation.csv", index=False)
    print("\\n=== PAPER TABLE 2 — ablations at h=10min ===")
    print(t2.to_string(index=False))
except Exception as e:
    print(f"Table 2 error: {e}")
'''

ZIP_DOWNLOAD_CODE = '''\
# ==== Zip and download all outputs ====
import shutil
zip_path = WORK_DIR / "solarsde_outputs_combined.zip"
if zip_path.exists(): zip_path.unlink()
print(f"Zipping {PERSIST_DIR} ...")
shutil.make_archive(str(zip_path).replace(".zip", ""), "zip", root_dir=PERSIST_DIR)
size_mb = zip_path.stat().st_size / 1e6
print(f"Archive: {zip_path}  ({size_mb:.1f} MB)")

if IN_COLAB:
    from google.colab import files
    try: files.download(str(zip_path))
    except Exception as e: print(f"Auto-download failed: {e}. File at {zip_path}")
else:
    print(f"On Kaggle: download via Output tab or from {zip_path}")

print("\\n" + "=" * 70)
print("ALL STAGES COMPLETE")
print("=" * 70)
for sub in ["splits", "extended", "checkpoints", "latents", "results", "figures"]:
    p = PERSIST_DIR / sub
    if p.exists():
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        print(f"  {sub}/: {n} files, {total/1e6:.1f} MB")
'''


# ================================================================
# Build
# ================================================================

def combined_nb():
    cells = [
        ("markdown", HEADER_MD),
        ("markdown", "## 0. Setup"),
        ("code", SETUP_CODE),
        ("code", FAST_START_CODE),
        ("markdown", "## 1. Shared code (metrics, SDE solver, models)"),
        ("code", SHARED_CODE),
        ("code", LOAD_DATA_CODE),
        ("markdown", "## STAGE A — Baselines (persistence, smart persistence, LSTM, MC-Dropout, CSDI)"),
        ("code", BASELINES_CODE),
        ("markdown", "## STAGE B — Ablations (no-CTI, no-score, no-SDE)"),
        ("code", ABLATIONS_CODE),
        ("markdown", "## STAGE C — Conformal Calibration"),
        ("code", CALIBRATION_CODE),
        ("markdown", "## STAGE D — CTI Analysis + Economic Value + Figures"),
        ("code", ANALYSIS_CODE),
        ("markdown", "## Final: zip & download"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    path = NB_DIR / "06_combined_baselines_ablations_analysis.ipynb"
    path.write_text(json.dumps(combined_nb(), indent=1))
    print(f"Wrote {path.name}: {path.stat().st_size / 1024:.1f} KB")
