# SolarSDE Colab Notebooks

5 self-contained notebooks that run the complete SolarSDE experimental pipeline on Google Colab or Kaggle. Each notebook fits within 8 hours on a free T4 / P100 GPU.

## Notebook overview

| # | File | Purpose | Runtime | Inputs | Outputs |
|---|------|---------|---------|--------|---------|
| 1 | `01_data_and_vae.ipynb` | Download NREL data, preprocess, train CS-VAE, extract latents + CTI | ~4-6 hr | None | `splits/`, `checkpoints/vae_*`, `latents/` |
| 2 | `02_sde_score_main.ipynb` | Train Neural SDE + Score Decoder, main evaluation | ~1-2 hr | Nb 1 outputs | `checkpoints/sde_*`, `score_*`, `solar_sde_main_results.csv` |
| 3 | `03_baselines.ipynb` | Train 5 baselines, combined results table | ~2-4 hr | Nb 1, 2 outputs | `main_results_combined.csv`, baseline CSVs |
| 4 | `04_ablations.ipynb` | 3 ablations (no-CTI, no-score, no-SDE) | ~3-5 hr | Nb 1, 2 outputs | `ablation_results.csv` |
| 5 | `05_analysis_figures.ipynb` | Multi-seed, CTI analysis, economic value, all figures | ~2-4 hr | Nb 1-4 outputs | `figures/*.pdf`, `paper_table*.csv`, analysis JSONs |

**Total runtime: ~12-20 hours of compute, split across 5 sessions.** Fits within Kaggle's 30 hours/week free quota.

## How to run

### On Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `01_data_and_vae.ipynb`.
2. `Runtime > Change runtime type > T4 GPU` (free tier is sufficient).
3. `Runtime > Run all`. The notebook will:
   - Prompt you to mount your Google Drive (authorize once).
   - Download data, preprocess, train the VAE, extract latents.
   - Save everything to `/content/drive/MyDrive/solarsde_outputs/`.
4. When complete, the last cell prints a summary and triggers a zip download.
5. Open Notebook 2 next — it reads from the same Drive folder, no re-upload needed.
6. Proceed through notebooks 3, 4, 5 in order.

### On Kaggle

1. Go to [kaggle.com/code](https://kaggle.com/code) and create a new notebook.
2. Upload the .ipynb file via `File > Upload notebook`.
3. Enable GPU: `Settings > Accelerator > GPU P100`.
4. Run all cells. Outputs go to `/kaggle/working/solarsde_outputs/`.
5. Commit the notebook — outputs are saved as a Kaggle dataset version.
6. For subsequent notebooks, add the previous notebook's output dataset as input.

## What each notebook logs to terminal

Every notebook prints:
- Environment info (Colab/Kaggle/local, GPU name, memory)
- Dataset sizes at each preprocessing step
- Per-epoch training losses (train, val, components)
- Evaluation metrics per horizon (CRPS, RMSE, MAE, PICP, PINAW, ramp CRPS)
- Summary statistics at the end (file listing, sizes)

**All terminal output is preserved** — if you save the notebook after running, all print statements become part of the saved file. Even if you lose the downloaded zip, the notebook itself contains the full experimental log.

## Outputs structure (in persistent storage)

```
solarsde_outputs/
├── splits/
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
├── checkpoints/
│   ├── vae_best.pt
│   ├── sde_best.pt
│   ├── score_best.pt
│   ├── sde_a2_best.pt        # ablation A2
│   ├── linear_decoder_a4.pt  # ablation A4
│   ├── sde_a5_best.pt        # ablation A5
│   ├── sde_seed{123,456}.pt  # multi-seed
│   └── score_seed{123,456}.pt
├── latents/
│   └── {split}_{latents,cti,ghi,covariates,is_ramp}.npy
├── results/
│   ├── vae_training_history.csv
│   ├── sde_training_history.csv
│   ├── score_training_history.csv
│   ├── solar_sde_main_results.csv
│   ├── main_results_combined.csv
│   ├── baseline_{name}_results.csv
│   ├── ablation_results.csv
│   ├── multi_seed_aggregated.csv
│   ├── cti_analysis.json
│   ├── economic_value.json
│   ├── reliability_data.json
│   ├── paper_table1_main.csv
│   └── paper_table2_ablation.csv
└── figures/
    ├── fig2_crps_vs_horizon.pdf
    ├── fig3a_cti_scatter.pdf
    ├── fig3b_crps_by_cti.pdf
    ├── fig5_reliability.pdf
    ├── fig6_economic_value.pdf
    └── fig_pit_histogram.pdf
```

## Checkpointing and recovery

- Each notebook saves its outputs to **Google Drive / Kaggle persistent storage** at every checkpoint.
- Outputs are also **zipped and auto-downloaded** at the end of every notebook.
- Training loops save the best checkpoint after every epoch — interrupting and restarting won't lose progress beyond the current epoch.
- If a session disconnects mid-way, re-running the notebook will resume from the last saved checkpoint (all training loops check for existing checkpoints).

## Minimum RAM / VRAM requirements

| Component | VRAM (peak) | RAM (peak) | Disk |
|-----------|-------------|------------|------|
| CS-VAE training | 3 GB | 6 GB | 5 GB (images) |
| Neural SDE / Score Decoder | <1 GB | 2 GB | <100 MB (latents) |
| Baselines | 1 GB | 2 GB | <100 MB |
| Ablations | 2 GB | 3 GB | ~200 MB |
| Analysis / figures | <500 MB | 2 GB | <100 MB |

Colab free T4 (16 GB VRAM, 12 GB RAM, 80 GB disk) handles all notebooks comfortably.

## Troubleshooting

- **Session disconnects during CS-VAE training (Notebook 1)** — simply re-run the notebook. It will detect existing VAE checkpoints and skip re-training.
- **Drive mount fails on Colab** — revoke permissions in Google Account settings and re-authorize, or fall back to manual upload via the file browser in the left sidebar.
- **"CUDA out of memory" on Kaggle T4 x2** — use the P100 accelerator option (single GPU, 16 GB) instead.
- **Missing prior notebook outputs** — run the notebooks in order; each one reads from the persistent storage written by its predecessor.
