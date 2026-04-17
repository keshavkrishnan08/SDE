# CLAUDE.md — SolarSDE: Neural Stochastic Differential Equations for Probabilistic Solar Irradiance Nowcasting from All-Sky Imagery

## PROJECT IDENTITY

**Title:** "Latent Neural Stochastic Differential Equations with Conditional Score Matching for Probabilistic Solar Irradiance Nowcasting"
**Target Venues (ranked):** (1) NeurIPS 2027 Main Conference (2) Nature Energy (3) Applied Energy (4) AAAI 2027 Main Conference
**Author:** Keshav Krishnan (Sole Author)
**Timeline:** 10–14 weeks from start to submission-ready manuscript
**Compute:** Single GPU (RTX 3090 / A100 equivalent); all experiments must be runnable on consumer hardware

---

## 1. EXECUTIVE SUMMARY

### 1.1 The Problem

Solar energy integration into electrical grids is bottlenecked by **irradiance intermittency uncertainty**. Grid operators maintain expensive fossil fuel spinning reserves because they cannot predict solar output at minute-scale resolution with calibrated uncertainty. Current forecasting methods are either:
- **Deterministic CNNs** on sky camera images → produce point forecasts with no uncertainty quantification
- **Ensemble methods** (Monte Carlo dropout, deep ensembles) → produce crude, physically uncalibrated uncertainty bands
- **Classical SDEs** → use hand-crafted drift/diffusion with no learned representations
- **Diffusion models for time series** (TimeGrad, CSDI, ScoreGrad) → operate in observation space, not in a learned physical manifold; treat all timesteps identically regardless of cloud state

**Gap:** No existing method learns a state-dependent stochastic process in a physically meaningful latent space where the noise amplitude is conditioned on the actual cloud dynamics extracted from imagery.

### 1.2 The Solution

**SolarSDE** — a three-component architecture:

1. **Cloud-State Variational Autoencoder (CS-VAE):** Encodes all-sky fisheye images into a low-dimensional latent manifold capturing cloud morphology, motion, and optical thickness. Produces a continuous "cloud state" trajectory z_t.

2. **Latent Neural SDE:** Evolves the cloud state z_t forward in time via learned drift μ_θ(z_t, t) and state-dependent diffusion σ_θ(z_t, t). The diffusion coefficient is explicitly conditioned on a scalar **Cloud Turbulence Index (CTI)** extracted from the latent dynamics — meaning the model "knows" that uncertainty should spike at cloud edges and collapse under uniform overcast or clear sky.

3. **Conditional Score-Matching Decoder:** Instead of a simple decoder, uses a conditional score-matching diffusion model to map from latent trajectories back to irradiance distributions. This produces calibrated probabilistic forecasts at arbitrary future timesteps without autoregressive rollout.

### 1.3 The Three Novelty Claims

**Novelty 1 (Architectural):** First Neural SDE where the diffusion coefficient is conditioned on a learned physical state variable (CTI) extracted from imagery, rather than being a free neural network. This is a physics-informed constraint on the stochastic process that improves calibration.

**Novelty 2 (Methodological):** First combination of Neural SDEs with conditional score-matching diffusion for any spatiotemporal forecasting problem. Neural SDEs model the latent dynamics; score matching generates the observation-space distribution. These two generative mechanisms have never been combined.

**Novelty 3 (Applied):** First probabilistic solar irradiance nowcasting system that produces state-dependent uncertainty quantification from raw sky imagery at sub-minute resolution. The CTI provides grid operators with a single, interpretable scalar indicating "how uncertain should we be right now."

---

## 2. DATASETS

### 2.1 Primary Dataset: CloudCV 10-Second Sky Images and Irradiance

- **Source:** NREL Solar Radiation Research Laboratory (SRRL), Golden, Colorado
- **URL:** https://data.nrel.gov/submissions/248
- **Contents:** Sky images (fisheye 180°) captured every 10 seconds + co-located LI-COR LI200 pyranometer irradiance measurements
- **Duration:** 90 days (September 5 – December 3, 2019)
- **Resolution:** 10-second intervals during daylight hours
- **Approximate size:** ~750,000 image-irradiance pairs
- **Format:** JPEG images + CSV irradiance readings
- **License:** Public domain (U.S. government work)

### 2.2 Secondary Dataset: SRRL Baseline Measurement System (BMS)

- **Source:** NREL SRRL BMS
- **URL:** https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS
- **Contents:** 130+ meteorological variables at 1-minute intervals since 1981
- **Key variables:** GHI (Global Horizontal Irradiance), DNI (Direct Normal Irradiance), DHI (Diffuse Horizontal Irradiance), ambient temperature, relative humidity, barometric pressure, wind speed/direction, cloud cover percentage from TSI-880
- **Use:** Extended irradiance ground truth, meteorological covariates for auxiliary features, and long-term validation
- **Duration:** 40+ years (use 2017–2024 for overlap with ASI-16 sky camera)
- **Format:** CSV via MIDC data download interface

### 2.3 Tertiary Dataset: NSRDB (National Solar Radiation Database)

- **Source:** NREL via Azure Blob Storage or API
- **URL:** https://nsrdb.nrel.gov/
- **Contents:** Half-hourly GHI/DNI/DHI at ~4km resolution, derived from GOES satellite + FARMS radiative transfer model
- **Use:** Multi-site generalization experiments (train on Golden CO, test on other locations); provides satellite-derived clear-sky index as additional covariate
- **Duration:** 1998–2023
- **Format:** HDF5 (.h5) files

### 2.4 Data Pipeline Specification

```
data/
├── raw/
│   ├── cloudcv/           # 10-second images + irradiance CSV
│   ├── bms/               # 1-minute BMS meteorological data
│   └── nsrdb/             # Half-hourly satellite-derived irradiance
├── processed/
│   ├── aligned/           # Images + irradiance temporally aligned
│   ├── sequences/         # Sliding window sequences for training
│   ├── clear_sky/         # Clear-sky model outputs (Ineichen-Perez)
│   └── splits/            # Train/val/test splits
└── metadata/
    ├── solar_geometry.csv  # Solar zenith/azimuth for each timestamp
    └── quality_flags.csv   # SERI-QC quality control flags
```

**Data Agent Tasks:**
1. Download CloudCV dataset from NREL data portal
2. Download overlapping BMS data for the same 90-day period
3. Compute clear-sky irradiance using Ineichen-Perez model (pvlib-python) for each timestamp
4. Compute clear-sky index: k_t = GHI_measured / GHI_clearsky
5. Compute solar zenith angle for each timestamp (pvlib-python)
6. Filter: remove nighttime (solar zenith > 85°), remove flagged quality data
7. Normalize images: crop to circular fisheye region, resize to 256×256, normalize pixel values [0,1]
8. Create sliding window sequences: input = {img_{t-T:t}, GHI_{t-T:t}}, target = GHI_{t+1:t+H}
9. Split: 60 days train / 15 days validation / 15 days test (chronological, no shuffle)
10. Create ramp event labels: |ΔGHI / Δt| > threshold (define threshold from literature: typically 50 W/m² per minute)

---

## 3. MODEL ARCHITECTURE

### 3.1 Component 1: Cloud-State Variational Autoencoder (CS-VAE)

**Purpose:** Learn a continuous low-dimensional representation of cloud morphology from fisheye sky images.

**Encoder E_φ:**
```
Input: x_t ∈ R^{256×256×3}  (RGB sky image)
│
├── Conv2D(3→32, k=4, s=2, p=1) + GroupNorm(8) + SiLU
├── Conv2D(32→64, k=4, s=2, p=1) + GroupNorm(16) + SiLU
├── Conv2D(64→128, k=4, s=2, p=1) + GroupNorm(32) + SiLU
├── Conv2D(128→256, k=4, s=2, p=1) + GroupNorm(64) + SiLU
├── Conv2D(256→512, k=4, s=2, p=1) + GroupNorm(128) + SiLU
├── AdaptiveAvgPool2D(1×1)
├── Flatten → Linear(512 → 2 × d_z)
│
Output: μ_z ∈ R^{d_z}, log_σ²_z ∈ R^{d_z}
        z_t = μ_z + σ_z ⊙ ε,  ε ~ N(0, I)
```

**Decoder D_ψ:**
```
Input: z_t ∈ R^{d_z}
│
├── Linear(d_z → 512 × 8 × 8) + Reshape(512, 8, 8)
├── ConvTranspose2D(512→256, k=4, s=2, p=1) + GroupNorm(64) + SiLU
├── ConvTranspose2D(256→128, k=4, s=2, p=1) + GroupNorm(32) + SiLU
├── ConvTranspose2D(128→64, k=4, s=2, p=1) + GroupNorm(16) + SiLU
├── ConvTranspose2D(64→32, k=4, s=2, p=1) + GroupNorm(8) + SiLU
├── ConvTranspose2D(32→3, k=4, s=2, p=1) + Sigmoid
│
Output: x̂_t ∈ R^{256×256×3}
```

**Hyperparameters:**
- Latent dimension d_z = 64 (ablate: 16, 32, 64, 128, 256)
- β-VAE weight β = 0.1 (ablate: 0.01, 0.1, 1.0, 4.0)
- Training: Adam, lr=1e-4, batch=64, 100 epochs
- Loss: L_VAE = ||x_t - x̂_t||² + β · KL(q(z|x) || p(z))

**Cloud Turbulence Index (CTI) Extraction:**

The CTI is computed from temporal dynamics in the latent space:

```python
# Given a window of latent states z_{t-W:t} (W = 10 frames = 100 seconds)
z_velocities = z[1:] - z[:-1]  # Δz/Δt in latent space
CTI_t = ||Var(z_velocities)||_2  # L2 norm of variance of latent velocity

# Interpretation:
# CTI ≈ 0: Stable sky (clear or uniformly overcast) → low uncertainty
# CTI >> 0: Rapidly changing cloud state → high uncertainty (cloud edges, broken clouds)
```

CTI is a SCALAR that summarizes "how turbulent is the cloud field right now" in a single number. This is the key innovation: it makes the diffusion coefficient of the Neural SDE physically interpretable.

### 3.2 Component 2: Latent Neural SDE

**Purpose:** Model the stochastic evolution of the cloud state z_t forward in time.

**Mathematical formulation:**
```
dz_t = μ_θ(z_t, t, c_t) dt + σ_θ(z_t, CTI_t) dW_t
```

where:
- z_t ∈ R^{d_z}: latent cloud state at time t
- μ_θ: drift network (deterministic component, expected trajectory)
- σ_θ: diffusion network (stochastic component, state-dependent noise)
- c_t: auxiliary meteorological covariates [solar_zenith, temperature, humidity, wind_speed, clear_sky_index]
- CTI_t: cloud turbulence index (scalar, from CS-VAE)
- W_t: standard Brownian motion in R^{d_z}

**Drift Network μ_θ:**
```
Input: [z_t ∈ R^{d_z}, t ∈ R^1, c_t ∈ R^{d_c}]
│
├── Concat → Linear(d_z + 1 + d_c → 256) + SiLU
├── Linear(256 → 256) + SiLU
├── ResBlock(256) + SiLU
├── ResBlock(256) + SiLU
├── Linear(256 → d_z)
│
Output: μ_θ(z_t, t, c_t) ∈ R^{d_z}
```

**Diffusion Network σ_θ (CTI-Conditioned):**

This is the core architectural innovation. The diffusion coefficient is NOT a free neural network — it is explicitly gated by the Cloud Turbulence Index:

```
Input: [z_t ∈ R^{d_z}, CTI_t ∈ R^1]
│
├── CTI_t → Linear(1 → 64) + Softplus → α_t ∈ R^{64}  [CTI gate]
├── z_t → Linear(d_z → 64) + SiLU → h_z ∈ R^{64}      [state features]
├── h_z ⊙ α_t → Linear(64 → d_z) + Softplus            [element-wise gating]
│
Output: σ_θ(z_t, CTI_t) ∈ R^{d_z}  (diagonal diffusion, all positive)
```

**Key design choice:** The Softplus on CTI ensures that when CTI ≈ 0 (stable sky), the gate α_t → small values, compressing the diffusion toward zero. When CTI is large (turbulent sky), α_t → large values, amplifying the noise. This encodes the physics: uncertainty comes from cloud turbulence, not from clear skies.

**SDE Solver:**
- Method: Euler-Maruyama (simple, sufficient for moderate step sizes)
- Step size: Δt = 10 seconds (matching data resolution)
- Forecast horizon H: 1 to 30 minutes (6 to 180 steps)
- Number of Monte Carlo sample paths per forecast: N = 100 (ablate: 50, 100, 200, 500)

**Training Method:**

Use **SDE Matching** (Bartosh et al., ICLR 2025) for simulation-free training. This avoids the computational cost of adjoint sensitivity methods:

```
L_SDE = E_t [||μ_θ(z_t, t, c_t) - (z_{t+Δt} - z_t)/Δt||²
         + λ_σ · ||σ_θ(z_t, CTI_t)² - (z_{t+Δt} - z_t - μ_θ·Δt)² / Δt||²]
```

The first term matches the drift to observed finite differences. The second term matches the diffusion to the residual variance. λ_σ = 1.0 (ablate: 0.1, 0.5, 1.0, 5.0).

### 3.3 Component 3: Conditional Score-Matching Irradiance Decoder (CSMID)

**Purpose:** Map from latent SDE trajectories to calibrated irradiance probability distributions.

**Why not a simple linear decoder?** A linear decoder z_t → GHI_t would produce Gaussian predictive distributions (because the SDE generates Gaussian-ish latent paths). Real irradiance distributions are highly non-Gaussian: they have sharp modes under clear sky, bimodal distributions under broken clouds, and heavy-tailed ramp events. Score-matching diffusion generates arbitrary distributions.

**Architecture:**

The CSMID operates as a conditional denoising diffusion model in 1D (irradiance space), conditioned on the latent state z_t:

```
# Forward diffusion (noise schedule):
GHI_s = √(ᾱ_s) · GHI_0 + √(1 - ᾱ_s) · ε,  ε ~ N(0, 1)
where s ∈ [0, 1] is the diffusion time (NOT physical time)

# Score network s_ω:
Input: [GHI_s ∈ R^1, s ∈ R^1, z_t ∈ R^{d_z}, CTI_t ∈ R^1, c_t ∈ R^{d_c}]
│
├── Concat all → Linear(1 + 1 + d_z + 1 + d_c → 256) + SiLU
├── ResBlock(256) + SiLU
├── ResBlock(256) + SiLU
├── Linear(256 → 1)
│
Output: ŝ_ω ≈ ∇_{GHI_s} log p_s(GHI_s | z_t, CTI_t, c_t)
```

**Training loss (denoising score matching):**
```
L_CSMID = E_{s, ε} [||ŝ_ω(GHI_s, s, z_t, CTI_t, c_t) + ε/√(1-ᾱ_s)||²]
```

**Inference (sampling):**
To generate a probabilistic forecast at time t+h:
1. Run the Neural SDE forward from z_t to z_{t+h} (N=100 sample paths)
2. For each sample path endpoint z_{t+h}^{(n)}, run reverse diffusion with CSMID to sample GHI_{t+h}^{(n)}
3. The ensemble {GHI_{t+h}^{(1)}, ..., GHI_{t+h}^{(N)}} IS the probabilistic forecast

**Noise schedule:** Linear β schedule, 100 diffusion steps (ablate: 50, 100, 200)

### 3.4 Full Training Pipeline

**Stage 1: Train CS-VAE (independent pre-training)**
- Input: individual sky images
- Loss: L_VAE = reconstruction + β · KL
- Duration: ~100 epochs
- Output: frozen encoder E_φ, frozen decoder D_ψ

**Stage 2: Extract latent trajectories + CTI**
- Run frozen E_φ on all training images → latent trajectory {z_t}
- Compute CTI for each timestep from latent velocity variance
- Save as preprocessed features

**Stage 3: Train Latent Neural SDE**
- Input: consecutive latent state pairs (z_t, z_{t+Δt}) with covariates
- Loss: L_SDE (drift matching + diffusion matching)
- Duration: ~200 epochs
- Output: trained drift μ_θ and diffusion σ_θ

**Stage 4: Train CSMID**
- Input: (GHI_t, z_t, CTI_t, c_t) pairs
- Loss: L_CSMID (denoising score matching)
- Duration: ~100 epochs
- Output: trained score network s_ω

**Stage 5: End-to-end fine-tuning (optional)**
- Unfreeze all components
- Joint loss: L_total = L_VAE + α · L_SDE + γ · L_CSMID
- α = 1.0, γ = 1.0 (ablate ratios)
- Duration: ~50 epochs with reduced learning rate (1e-5)

---

## 4. BASELINES (7 models)

### 4.1 Persistence Model (Naive Baseline)
- Forecast: GHI_{t+h} = GHI_t (last observed value persists)
- Probabilistic version: add Gaussian noise calibrated from training set residuals

### 4.2 Smart Persistence
- Forecast: k_{t+h} = k_t (persist the clear-sky index, then multiply by clear-sky model)
- Standard baseline in solar forecasting literature

### 4.3 LSTM Deterministic
- Standard LSTM on irradiance + meteorological time series
- Point forecast only (no uncertainty)
- Architecture: 2-layer LSTM, hidden=128, lookback=30 steps

### 4.4 LSTM + Monte Carlo Dropout (MC-Dropout)
- Same LSTM with dropout=0.1 at inference
- Run 100 forward passes → ensemble → probabilistic forecast
- Standard uncertainty quantification baseline

### 4.5 Deep Ensemble (5 LSTMs)
- Train 5 LSTMs with different random seeds
- Ensemble predictions → probabilistic forecast
- Strongest non-generative probabilistic baseline

### 4.6 TimeGrad (Rasul et al., ICML 2021)
- Autoregressive diffusion model for time series
- Uses RNN encoder + DDPM decoder
- State-of-the-art diffusion baseline for probabilistic forecasting

### 4.7 CSDI (Tashiro et al., NeurIPS 2021)
- Conditional Score-based Diffusion for Imputation/forecasting
- Non-autoregressive, transformer-based
- State-of-the-art score-based baseline

### 4.8 CNN + Sky Image (Deterministic Vision Baseline)
- ResNet-18 encoder on sky image → Linear → GHI prediction
- Tests whether images alone contain predictive information
- Point forecast only

---

## 5. EXPERIMENTS

### 5.1 Main Experiment: Probabilistic Forecasting Performance

**Setup:**
- Forecast horizons: h ∈ {1, 2, 5, 10, 15, 20, 30} minutes
- All models trained on same train split, evaluated on same test split
- N=100 Monte Carlo samples for all probabilistic methods

**Metrics:**

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| CRPS | ∫(F(y) - 1{y ≤ x})² dy | Overall probabilistic calibration (lower = better) |
| PICP | % of observations inside 90% PI | Prediction interval coverage (target: 90%) |
| PINAW | Mean width of 90% PI / range | Sharpness of intervals (lower = better, given correct coverage) |
| RMSE | √(E[(ŷ - y)²]) | Point forecast accuracy (median of ensemble) |
| MAE | E[|ŷ - y|] | Point forecast accuracy |
| Ramp Score | CRPS conditional on |ΔGHI| > 50 W/m²/min | Performance specifically during ramp events |
| Skill Score | 1 - CRPS_model / CRPS_persistence | Improvement over persistence (higher = better) |
| Reliability Diagram | Observed coverage vs. nominal coverage at multiple levels | Visual calibration assessment |
| QQ Plot | Quantile-quantile of PIT values | Distributional calibration |

### 5.2 Ablation Study (7 rows)

Each ablation removes or replaces ONE component to isolate its contribution:

| Row | Model Variant | What's Changed | Tests |
|-----|---------------|----------------|-------|
| A1 | SolarSDE (full) | Nothing (reference) | Full system performance |
| A2 | SolarSDE − CTI gating | σ_θ(z_t) without CTI conditioning | Is the CTI gating mechanism necessary? |
| A3 | SolarSDE − CS-VAE (raw pixels) | Neural SDE operates on PCA-reduced pixel features | Is the learned latent space better than raw features? |
| A4 | SolarSDE − Score Matching | Replace CSMID with linear decoder z→GHI | Is score-matching necessary for non-Gaussian distributions? |
| A5 | SolarSDE − Neural SDE (deterministic ODE) | Set σ_θ = 0, use Neural ODE for latent dynamics | Is stochastic modeling necessary? (tests aleatoric uncertainty) |
| A6 | SolarSDE − SDE Matching (adjoint training) | Train with adjoint sensitivity instead of SDE Matching | Is SDE Matching training necessary for performance? |
| A7 | SolarSDE − Meteorological covariates | Remove c_t from drift network | Do meteorological features help beyond imagery? |

### 5.3 CTI Analysis Experiment

**Purpose:** Validate that the learned CTI is physically meaningful.

**Tests:**
1. **CTI vs. observed cloud cover:** Compute Spearman correlation between CTI and TSI-880 measured cloud cover percentage. Expected: high positive correlation.
2. **CTI vs. irradiance variability:** Compute correlation between CTI and rolling standard deviation of GHI (1-minute window). Expected: high positive correlation.
3. **CTI vs. forecast error:** Plot model CRPS as a function of CTI quantile bins. Expected: CRPS increases monotonically with CTI.
4. **CTI regime analysis:** K-means cluster CTI values into 4 regimes (clear, thin cloud, broken cloud, overcast). Show each regime has distinct irradiance distribution shape.
5. **Visualization:** Side-by-side: sky image → latent space (t-SNE) → CTI value → predicted distribution shape. Show 4 representative examples (one per regime).

### 5.4 Ramp Event Detection Experiment

**Purpose:** Evaluate performance specifically during irradiance ramp events, which are the operationally critical events for grid operators.

**Definition:** A ramp event occurs when |ΔGHI| > 50 W/m² within a 1-minute window.

**Metrics:**
- Ramp detection probability: P(model predicts ramp | ramp occurs)
- Ramp false alarm rate: P(model predicts ramp | no ramp)
- Ramp detection lead time: How many minutes before the ramp does the model first assign >50% probability?
- AUROC for ramp/no-ramp classification (using the width of the predictive interval as the decision variable: wide interval → predict ramp)

### 5.5 Economic Value Experiment

**Purpose:** Quantify the dollar value of improved probabilistic forecasts for grid operation.

**Method:**
1. Simulate a grid operator making reserve commitment decisions every 5 minutes
2. The operator must hold enough spinning reserve to cover the (1-α) quantile of the predictive distribution (α = 0.05, i.e., 95% reliability)
3. Cost = (reserve capacity held) × (marginal reserve cost, ~$50/MWh from CAISO data)
4. Penalty for under-reserve = (shortfall) × (penalty rate, ~$1000/MWh)
5. Compute total cost for each forecasting model over the test period
6. Report annual savings: ΔCost = Cost_baseline - Cost_SolarSDE, extrapolated to a 1 GW solar plant

### 5.6 Generalization Experiment (Multi-Site)

**Purpose:** Test whether SolarSDE trained on Golden, CO generalizes to other locations.

**Method:**
1. Use NSRDB satellite-derived irradiance for 5 geographically diverse locations
2. Fine-tune the pre-trained SolarSDE on each location's data
3. Report CRPS and skill score at each location
4. Compare zero-shot (no fine-tuning) vs. fine-tuned performance

### 5.7 Latent Space Interpretability Experiment

**Purpose:** Show that the CS-VAE has learned physically meaningful representations.

### 5.8 Sampling Efficiency Experiment

**Purpose:** Characterize how many Monte Carlo samples N are needed for converged probabilistic forecasts.

---

## 6. STATISTICAL RIGOR

- **Diebold-Mariano test** for pairwise forecast comparison
- **Bootstrap confidence intervals** (B=1000) for all reported metrics
- **Holm-Bonferroni correction** for multiple comparisons
- **PIT histograms** and **reliability diagrams** for calibration
- Fix random seeds (42, 123, 456) for 3 independent runs

---

## 7. KEY IMPLEMENTATION NOTES

1. **Start with data.** Download and preprocess BEFORE touching any model code.
2. **Train CS-VAE first and freeze it.**
3. **CTI is derived, not learned.** Deterministic function of the latent trajectory.
4. **The diffusion coefficient must always be positive.** Use Softplus activations.
5. **SDE Matching is CRITICAL.** Do NOT use adjoint sensitivity as primary training method.
6. **Score-matching decoder operates in 1D.** ~50K params. Trains in minutes.
7. **All baselines must use the SAME data splits and evaluation pipeline.**
8. **Always work with clear-sky index k_t = GHI/GHI_clear.**
9. **Ramp events are RARE.** ~5-10% of timesteps.
10. **Solar zenith angle filtering.** Remove data with zenith > 85°.
