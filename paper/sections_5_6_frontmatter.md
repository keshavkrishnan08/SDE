# Front matter + Sections 5–6 + Appendix C

## Nomenclature
| symbol | meaning |
|---|---|
| *P*ₜ | PV power at minute *t* (kW) |
| *x*ₜ | all-sky image at minute *t* (64×64×3) |
| *C*(*m*,τ) | clear-sky-PV envelope (month *m*, minute-of-day τ) |
| *k*ₜ | clear-sky index, *P*ₜ/*C* |
| *δk*ₜ₊ₕ | clear-sky-index residual *k*ₜ₊ₕ − *k*ₜ |
| *z*ₜ | cloud-state latent (ℝ⁶⁴) |
| CTI | Cloud Turbulence Index |
| *h* | forecast horizon (min) |
| *L* | history length (16 min) |
| *K* | number of OU mixture components (3) |
| π, μ, θ, σ | mixture weights, means, mean-reversion rates, volatilities |
| *w* | persistence-blend weight |
| *s*ₕ | conformal scale at horizon *h* |
| Φ, φ | standard normal cdf, pdf |
| *Abbreviations:* | CRPS, PICP, PINAW, PIT, DM (Diebold–Mariano), VAE, SDE, OU (Ornstein–Uhlenbeck), CV |

## 5. Discussion

**Stochastic-process characterization, not image prediction.** The prevailing image-based paradigm predicts the future sky and regresses power from it. SolarSDE reframes nowcasting as characterizing the stochastic process that governs the cloud state: a single learned latent SDE yields calibrated marginals at every horizon in closed form. This avoids generating and then discarding high-dimensional pixels, avoids error compounding through a generative rollout, and produces a contiguous 1–30 minute forecast band rather than a single lead time. The Cloud Turbulence Index makes the stochasticity interpretable — a scalar an operator can read as "how trustworthy is the forecast right now."

**Calibration is the operative property.** Our central empirical message is that an image-based probabilistic forecaster can be made *verifiably* calibrated. Where prior work optimizes sharpness and leaves coverage unaudited, SolarSDE reports PICP near nominal, PIT consistent with uniformity, and reliability diagrams on the diagonal. For reserve scheduling this matters more than a marginally lower CRPS at one horizon: an interval that covers as claimed is what determines whether the held reserve is correct.

**On the cloudy-day comparison.** On SkyGPT's deliberately hard cloudy test, SolarSDE is competitive (CRPS 3.03 vs 2.81) while being calibrated, multi-horizon, and three orders of magnitude smaller. We attribute the residual gap to the information bottleneck of a globally-pooled cloud-state latent, which retains cloud appearance but not fine spatial position; a generative model that synthesizes the full future frame can exploit the latter. Bridging this without sacrificing calibration or efficiency is the natural next step.

**Deployability.** At 0.43 M parameters and ≈4 ms per forecast, SolarSDE runs at the edge — beside the inverter at distributed PV assets — where a generative video pipeline cannot. For the distributed-solar setting this is a qualitative, not incremental, difference.

## 6. Conclusions
We introduced SolarSDE, a cloud-turbulence-gated latent neural stochastic differential equation for probabilistic PV nowcasting. By modelling the clear-sky-index residual as the closed-form marginal of a CTI-gated mixture of Ornstein–Uhlenbeck processes, anchored to persistence and calibrated by group-conditional conformal prediction, it delivers sharp, verifiably-calibrated forecasts across the full 1–30 minute operational band from a single 0.43 M-parameter model. On the SKIPP'D benchmark it outperforms persistence, smart persistence, LSTM, MC-Dropout and CSDI at every horizon with Diebold–Mariano significance, exhibits well-calibrated coverage and physically-meaningful uncertainty, generates positive reserve value across all tested price scenarios, and is competitive with the heavyweight generative state of the art at its single 15-minute horizon. The result establishes the first calibration-verified, full-band intra-hour probabilistic PV nowcaster, and argues for treating nowcasting as stochastic-process characterization rather than future-image prediction.

## Appendix C — Implementation and reproducibility
**Architecture.** CS-VAE: 4-layer conv encoder/decoder (32→256 ch, GroupNorm, SiLU), 64-d latent, β=0.01, 12 epochs, AdamW 1e-3. Forecaster: 2-layer transformer encoder (d=128, 4 heads, GELU, pre-norm) over the 16-step history; per-component linear heads for π/μ/θ/σ; persistence-blend head; CTI diffusion gate. 0.43 M parameters total.

**Training.** AdamW (lr 5e-4, weight decay 1e-4), cosine schedule to 1e-5, 60 epochs, batch 128, gradient clip 1.0. Loss: closed-form Gaussian-mixture CRPS on *δk*. Ramp and high-CTI anchors oversampled. Seeds 42/123/456.

**Calibration.** Per-(horizon, CTI-quartile) conformal scale chosen on validation to minimize CRPS subject to PICP ≥ 0.88; scales applied multiplicatively at inference.

**Data.** SKIPP'D, public at purl.stanford.edu/dj417rh1007 (Nie et al., 2023). Chronological 70/15/15 day split. SkyGPT exact cloudy test from the SkyGPT release (Nie et al., 2024a). Clear-sky-PV envelope = month×minute-of-day 0.92 quantile, ±15-min rolling-max smoothed.

**Code/availability.** All code, configuration, and trained checkpoints released at github.com/keshavkrishnan08/SDE; experiments reproducible end-to-end from the provided notebooks on a single consumer GPU (~3–5 h).

## Declarations
- **CRediT:** K. Krishnan — conceptualization, methodology, software, validation, formal analysis, writing.
- **Competing interest:** none.
- **Generative AI:** AI assistance was used for code scaffolding and manuscript drafting; all results, analyses, and scientific claims are the author's.
- **Data availability:** SKIPP'D is public (link above); code and checkpoints released.
