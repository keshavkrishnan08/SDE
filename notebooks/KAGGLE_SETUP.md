# SolarSDE on Kaggle — Setup Guide

Step-by-step to run the combined notebook on Kaggle's free P100 GPU (30 hrs/week).
**No Google Drive needed** — Kaggle uses its own storage.

---

## 1. Create the Kaggle notebook

1. Go to https://www.kaggle.com/code → **New Notebook**
2. In the right sidebar:
   - **Accelerator:** `GPU P100` (free tier)
   - **Persistence:** `Files only`
   - **Environment:** `Always use latest environment`
   - **Internet:** `On` (required for GitHub pulls — must enable in your Kaggle phone-verified account)

## 2. Load the notebook

**Option A — Copy-paste (simplest):**

1. In your new Kaggle notebook, click **File → Import Notebook → URL**
2. Paste: `https://github.com/keshavkrishnan08/SDE/blob/main/notebooks/06_combined_baselines_ablations_analysis.ipynb`
3. Click **Import**

**Option B — Clone the repo:**

Add this as the first cell:

```python
!git clone --depth 1 https://github.com/keshavkrishnan08/SDE.git /kaggle/working/repo
%cd /kaggle/working/repo/notebooks
```

Then **File → Open → 06_combined_baselines_ablations_analysis.ipynb** from the file browser.

## 3. Enable internet (one-time step)

Kaggle requires phone verification to enable internet in notebooks. Go to https://www.kaggle.com/settings/account → **Phone Verification**. Without this, the fast-start can't pull data from GitHub.

## 4. Clear stale outputs (ONLY IF you've run this before)

If this is your **first Kaggle run**, skip — nothing to clear.

If you've run it before and want v2 from scratch, add this cell right after the setup cell:

```python
# Clean slate: remove old v1 checkpoints + results so Stage 0 retrains
import shutil
from pathlib import Path
P = Path("/kaggle/working/solarsde_outputs")
for f in ["checkpoints/sde_best.pt", "checkpoints/sde_final.pt",
          "checkpoints/score_best.pt", "checkpoints/score_final.pt",
          "checkpoints/sde_a2_best.pt", "checkpoints/sde_a5_best.pt",
          "checkpoints/linear_decoder_a4.pt",
          "results/solar_sde_main_results.csv",
          "results/main_results_combined.csv",
          "results/ablation_results.csv",
          "results/solar_sde_calibrated.csv"]:
    (P / f).unlink(missing_ok=True)
print("Cleared v1 artifacts — Stage 0 will retrain with v2 architecture.")
```

Run this cell once, then comment it out for subsequent re-runs (so resume-skip works).

## 5. Run it

Click **Run All** (top-right). The notebook will:

1. Auto-detect Kaggle and save outputs to `/kaggle/working/solarsde_outputs/`
2. Pull your trained VAE + latents from GitHub (~1 min, ~30 MB)
3. **Stage 0:** Train Neural SDE + Score Decoder (~35 min)
4. **Stage A:** Train 5 baselines (~2.5 hr)
5. **Stage B:** Train 3 ablations (~1 hr)
6. **Stage C:** Conformal calibration (~20 min)
7. **Stage D:** CTI analysis + all figures (~15 min)
8. Zip and place archive at `/kaggle/working/solarsde_outputs_combined.zip`

**Total: ~4.5 hours.** Well within Kaggle's 9-hour session limit.

## 6. Retrieving outputs

Kaggle gives you 3 ways to keep results:

**A. Commit the notebook** (RECOMMENDED — don't skip)

Before closing the notebook session:
- Click **Save Version** (top-right)
- Choose **Save & Run All (Commit)**
- This creates a snapshot with all outputs attached to your Kaggle account permanently

**B. Download the zip manually**

After completion, click the `/kaggle/working/solarsde_outputs_combined.zip` file in the right sidebar → the three-dots menu → **Download**.

**C. Create a Kaggle Dataset from the outputs**

In the notebook:
- Right sidebar → **Output** → "Create Dataset"
- Name it `solarsde-outputs` or similar
- This makes the files persistent and shareable

## 7. If session disconnects

Kaggle sessions time out after ~20 minutes of inactivity or 9 hours max runtime.

**Before disconnect:** if you see the kernel becoming unresponsive, click **Save Version → Save & Run All**. Then close and re-open; your outputs persist.

**After disconnect:** re-open the notebook, verify the outputs folder is still in `/kaggle/working/solarsde_outputs/` (Kaggle keeps working directory between runs in the same session). Re-run cells — each stage checks if its output exists and skips.

If `/kaggle/working/` was wiped (can happen after a Commit):
- Re-run from the top
- Fast-start will re-pull VAE + latents from GitHub
- Stage 0 will retrain (~35 min) — OR
- If you previously pushed checkpoints to GitHub, pull them into `/kaggle/working/solarsde_outputs/checkpoints/` manually

## 8. Known Kaggle gotchas

| Issue | Fix |
|-------|-----|
| `OSError: Could not connect to GitHub` | Internet not enabled — see step 3 |
| Stuck at `pip install` | Kaggle's package cache is slow; give it 2-3 min |
| `CUDA OOM` | Restart kernel, reduce `N_SAMPLES` from 50 to 25 in the notebook |
| Permission denied writing to `/kaggle/input/` | That folder is read-only. Use `/kaggle/working/` |
| Session killed unexpectedly | Check the blue "Session" indicator — may have been idle-killed. Commit + resume |

## 9. Running Notebooks 1-5 separately (not recommended on Kaggle)

If you want to run the 5-notebook pipeline instead of the combined one:
- Each notebook has the same Kaggle auto-detection
- You'd need to commit between them so outputs persist to a dataset
- Combined notebook (06) is strictly easier on Kaggle

---

## Quick command reference

```python
# Check what's in persistent storage
import os; os.listdir("/kaggle/working/solarsde_outputs")

# Force re-download of a specific file
!rm /kaggle/working/solarsde_outputs/checkpoints/vae_best.pt
# Then re-run fast-start cell

# Peek at final results mid-run
import pandas as pd
pd.read_csv("/kaggle/working/solarsde_outputs/results/main_results_combined.csv")
```

---

## Expected final output

After the full run, you'll have:

```
/kaggle/working/solarsde_outputs/
├── splits/         3 parquet files (train/val/test)
├── extended/       3 parquet files (90-day BMS)
├── checkpoints/    VAE + SDE + Score + baseline + ablation checkpoints
├── latents/        17 npy files per split
├── results/
│   ├── main_results_combined.csv          ← paper Table 1
│   ├── ablation_results.csv               ← paper Table 2
│   ├── solar_sde_calibrated.csv           ← raw vs calibrated metrics
│   ├── cti_analysis.json
│   ├── economic_value.json
│   └── reliability_data.json
└── figures/        6 PDF figures
```

Send me `results/main_results_combined.csv` and I'll verify v2 beat persistence.
