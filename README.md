# SolarSDE

**Latent Neural Stochastic Differential Equations with Conditional Score Matching for Probabilistic Solar Irradiance Nowcasting**

Keshav Krishnan

## Overview

SolarSDE is a three-component probabilistic solar irradiance nowcasting system:

1. **Cloud-State VAE (CS-VAE):** Encodes fisheye sky images into a latent cloud manifold
2. **Latent Neural SDE:** Evolves cloud states forward with CTI-gated diffusion
3. **Score-Matching Decoder (CSMID):** Maps latent trajectories to calibrated irradiance distributions

The key innovation is the **Cloud Turbulence Index (CTI)** — a scalar derived from latent dynamics that gates the SDE's diffusion coefficient, so uncertainty adapts to the physical state of the sky.

---

## Run on Colab (recommended)

Each notebook runs end-to-end on free Colab T4 GPU. Click a badge to open directly in Colab:

| # | Notebook | Runtime | Open |
|---|----------|---------|------|
| 1 | Data & CS-VAE | ~4-6 hr | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keshavkrishnan08/SDE/blob/main/notebooks/01_data_and_vae.ipynb) |
| 2 | SDE + Score + Main Eval | ~1-2 hr | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keshavkrishnan08/SDE/blob/main/notebooks/02_sde_score_main.ipynb) |
| 3 | Baselines | ~2-4 hr | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keshavkrishnan08/SDE/blob/main/notebooks/03_baselines.ipynb) |
| 4 | Ablations | ~3-5 hr | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keshavkrishnan08/SDE/blob/main/notebooks/04_ablations.ipynb) |
| 5 | Analysis & Figures | ~2-4 hr | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keshavkrishnan08/SDE/blob/main/notebooks/05_analysis_figures.ipynb) |

See [`notebooks/README.md`](notebooks/README.md) for detailed execution instructions, output schema, and troubleshooting.

Total runtime: ~12-20 hours of compute, split across 5 sessions. Fits within Kaggle's free 30-hour weekly quota.

---

## Local development

```bash
git clone git@github.com:keshavkrishnan08/SDE.git
cd SDE
pip install -r requirements.txt
python -m pytest tests/ -v    # 43 tests
```

Run the full pipeline locally (requires GPU):
```bash
python scripts/download_data.py       # ~10 min, ~2.7 GB
python scripts/preprocess_data.py     # ~1 min
python scripts/train_pipeline.py      # requires CUDA
```

## Project layout

```
.
├── CLAUDE.md                    Full project specification
├── README.md                    This file
├── requirements.txt             Python dependencies
├── configs/                     YAML configs (default + 6 ablations)
├── notebooks/                   5 Colab-ready notebooks
│   ├── 01_data_and_vae.ipynb
│   ├── 02_sde_score_main.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_ablations.ipynb
│   ├── 05_analysis_figures.ipynb
│   └── README.md
├── src/
│   ├── data/                    Download, preprocessing, datasets
│   ├── models/
│   │   ├── cs_vae.py            Cloud-State VAE
│   │   ├── cti.py               Cloud Turbulence Index
│   │   ├── neural_sde.py        Latent Neural SDE (CTI-gated diffusion)
│   │   ├── sde_solver.py        Euler-Maruyama solver
│   │   ├── score_decoder.py     Conditional score-matching decoder (CSMID)
│   │   ├── solar_sde.py         Full pipeline wrapper
│   │   └── baselines/           8 baseline implementations
│   ├── training/                5-stage training pipeline
│   ├── evaluation/              CRPS, PICP, PINAW, statistical tests
│   ├── visualization/           Figure generation
│   └── utils/                   Config, logging, seeding, I/O
├── scripts/                     Pipeline scripts (download, preprocess, train)
├── tests/                       43 unit tests (all passing)
└── paper/
    ├── main.tex                 10-page manuscript
    ├── supplementary.tex
    └── references.bib
```

## Data

Primary: [NREL CloudCV](https://data.openei.org/submissions/8294) — 10-second sky images + co-located irradiance from Golden, CO (Sep 5 – Dec 3, 2019). 8 days of high-quality sky imagery are currently publicly available (~2.6 GB).

Secondary: NREL SRRL BMS — 1-minute meteorological variables (GHI, DNI, DHI, temperature, humidity, wind, pressure, cloud cover) for the full 90-day period (~177 MB).

Notebook 1 auto-downloads both. Data files are excluded from git via `.gitignore` to keep the repo lean.

## Tests

```bash
python -m pytest tests/ -v
```

43 tests covering VAE, Neural SDE, Score Decoder, metrics, and data pipeline.

## Citation

If you use this work, please cite:

```bibtex
@misc{krishnan2026solarsde,
  title  = {Latent Neural Stochastic Differential Equations with Conditional
            Score Matching for Probabilistic Solar Irradiance Nowcasting},
  author = {Keshav Krishnan},
  year   = {2026},
  howpublished = {Manuscript in preparation}
}
```

## License

Research use. See LICENSE file (to be added).
