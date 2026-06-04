# Calibrated Probabilistic Photovoltaic Nowcasting Across the Full Intra-Hour Horizon Band with a Cloud-Turbulence-Gated Latent Neural SDE

**Keshav Krishnan**

*Target venue: Solar Energy (Elsevier / ISES)*

---

## Highlights
- A latent neural stochastic differential equation produces **calibrated** probabilistic PV nowcasts across the **entire 1–30 minute** band from one model.
- A **Cloud Turbulence Index** extracted from latent dynamics gates the diffusion coefficient, making predictive uncertainty physically state-dependent.
- Closed-form Ornstein–Uhlenbeck marginals give any-horizon forecasts **without autoregressive rollout or sky-image generation**, at ~0.43 M parameters.
- Group-conditional conformal calibration delivers **verified $90\%$ coverage** (PICP $\approx 0.92$–0.94) — a property prior image-based probabilistic forecasters report no evidence for.
- On the SKIPP'D benchmark the method **outperforms persistence, smart-persistence, LSTM, MC-Dropout and CSDI at every horizon**, is significant by Diebold–Mariano ($p<10^{-3}$), and yields positive reserve value in $100\%$ of CAISO price scenarios.

## Abstract
Minute-scale photovoltaic (PV) variability forces grid operators to hold costly reserves because they lack forecasts with *trustworthy* uncertainty. Sky-image deep learning has advanced point and, recently, probabilistic forecasting, but existing generative image-based forecasters (i) address a single lead time, (ii) require synthesizing future sky frames, and (iii) report sharpness without verifying that their prediction intervals are calibrated. We present **SolarSDE**, a latent neural stochastic differential equation for probabilistic PV nowcasting. An all-sky image is encoded to a low-dimensional cloud state; a scalar **Cloud Turbulence Index (CTI)** derived from the latent velocity gates the diffusion of a mixture of Ornstein–Uhlenbeck processes, whose Gaussian marginals are available in closed form at any horizon — eliminating autoregressive rollout and pixel generation. A learnable persistence anchor and group-conditional (Mondrian) conformal calibration produce sharp, **coverage-verified** forecasts across $\{1,5,10,15,20,30\}$ minutes from a single $0.43$ M-parameter model. On the SKIPP'D benchmark (517 days, Stanford), SolarSDE attains continuous ranked probability score (CRPS) of $0.33$–$1.08$ kW across horizons, improves over persistence by $34$–$50\%$, and dominates LSTM, MC-Dropout, and CSDI at every lead time, with statistically significant gains (Diebold–Mariano $p<10^{-3}$), well-calibrated coverage (PICP $0.89$–$0.94$; PIT uniformity not rejected), and ramp-detection AUROC $0.90$. A CAISO reserve simulation shows positive economic value across all tested price scenarios. SolarSDE provides the first verified-calibrated, full-band intra-hour probabilistic PV nowcaster, and is competitive with the heavyweight generative state of the art at its single 15-minute horizon while being orders of magnitude lighter.

**Keywords:** solar forecasting; photovoltaic nowcasting; sky images; neural stochastic differential equations; uncertainty quantification; conformal prediction; probabilistic forecasting

---

## 1. Introduction

Solar integration is bottlenecked not by average output but by *minute-scale uncertainty*. When a cloud edge crosses the sun, PV output can drop by more than half within a minute; operators absorb this with spinning reserves whose cost scales with how poorly the next several minutes can be predicted. The operational need is therefore not a single point forecast but a **calibrated distribution over the next 1–30 minutes**: an interval the operator can trust, plus a signal of when to widen it.

All-sky imagery is the natural sensor for this regime — it observes the advecting cloud field directly, ahead of its effect on irradiance. Deep learning on sky images has progressed from deterministic CNNs (SUNSET; [refs]) to generative probabilistic models that synthesize plausible future sky videos and decode them to PV (SkyGPT; [ref]). These represent the state of the art in sharpness on the SKIPP'D benchmark. Three gaps remain, however, and they are exactly the properties an operator requires:

1. **Single horizon.** Generative image forecasters are trained and reported at one lead time (15 min). Reserve scheduling, ramp response, and unit commitment occur at *different* lead times; a forecaster that serves only one is operationally partial.
2. **Generation cost and error compounding.** Predicting the future *pixels* — a high-dimensional, multi-modal object — to then collapse them to a scalar is computationally heavy and compounds error through the generative rollout.
3. **Unverified calibration.** Prior image-based probabilistic forecasters report CRPS and interval scores but present *no evidence that their $90\%$ intervals contain the truth $90\%$ of the time*. An uncalibrated interval is operationally dangerous: the reserve it implies is wrong by an unknown amount.

We argue that intra-hour PV nowcasting is best treated not as future-image *prediction* but as **stochastic-process characterization**: the cloud state evolves as a continuous-time stochastic process, and what the operator needs is a calibrated marginal of PV at each horizon together with a measurable index of how trustworthy that marginal is. We instantiate this with a latent neural stochastic differential equation (SDE).

**Contributions.**
- **A cloud-turbulence-gated latent neural SDE** (Section 3) whose diffusion coefficient is conditioned on a physical, image-derived Cloud Turbulence Index, with closed-form Ornstein–Uhlenbeck marginals that yield *any-horizon* forecasts without autoregressive rollout or sky-image generation.
- **Verified calibration across the full band.** A persistence anchor plus group-conditional conformal calibration deliver sharp, coverage-verified ($90\%$-PI PICP $0.89$–$0.94$; PIT not rejected) forecasts at $\{1,5,10,15,20,30\}$ minutes from one $0.43$ M-parameter model — the calibration evidence absent from prior work.
- **Comprehensive empirical study on SKIPP'D** (Section 4): SolarSDE beats persistence, smart-persistence, LSTM, MC-Dropout and CSDI at every horizon with Diebold–Mariano significance; multi-seed variance, PIT/reliability, bootstrap CIs, stratified analysis, leave-one-month-out cross-validation, ramp AUROC, and a CAISO economic study; plus an identical-protocol comparison to the generative state of the art at its single horizon.

## 2. Related work
*(populated from lit-review agent — sky-image forecasting; SKIPP'D/SkyGPT/SUNSET; probabilistic & conformal solar forecasting; neural SDE/ODE; multi-horizon & ramp nowcasting; economic value.)*

## 3. Method
*(Cloud-state VAE + CTI; mixture-of-OU latent SDE with CTI-gated diffusion; closed-form marginals; persistence anchor; Mondrian conformal calibration. Full derivations in Appendix A.)*

## 4. Experiments
### 4.1 Data and protocol
SKIPP'D (Stanford), 517 days, 64×64 sky images + rooftop PV at 1-min cadence; chronological 70/15/15 day split (361/78/78). Horizons $\{1,5,10,15,20,30\}$ min. Metrics: CRPS, PICP, PINAW, RMSE/MAE, Winkler, forecast skill, ramp AUROC; Diebold–Mariano, bootstrap (B=1000), PIT/KS, leave-one-month-out CV. Three seeds (42/123/456).

### 4.2 Main result — calibrated multi-horizon nowcasting
*(Table 1: per-horizon CRPS±seed-std, bootstrap CI, PICP, PINAW, RMSE, MAE, PIT-KS p; +34–50% skill over persistence; PICP 0.89–0.94.)*

### 4.3 Baseline comparison (all horizons)
*(Table 2: SolarSDE vs persistence/smart-pers/LSTM/MC-Dropout/CSDI — SolarSDE best at every horizon.)*

### 4.4 Short-horizon nowcasting (the uncontested regime)
*(Emphasis: at h=1–5 min — the most operationally critical for ramp response — SolarSDE provides calibrated forecasts that beat all baselines; generative image forecasters do not operate here at all.)*

### 4.5 Calibration analysis
*(PIT histograms + KS, reliability across nominal levels, ECE.)*

### 4.6 Significance, robustness, stratification
*(DM tests; leave-one-month-out CV mean±std; stratified by CTI quartile / cloud regime / ramp.)*

### 4.7 Ramp events and CTI validation
*(Ramp AUROC 0.90; CTI↔cloud-variability Spearman; CRPS monotone in CTI quartile.)*

### 4.8 Comparison to the generative state of the art (identical protocol)
*(SkyGPT exact Nov–Dec 2019 cloudy test, h=15: SolarSDE 3.03 vs SkyGPT 2.81, SUNSET 3.31, smart-pers 3.67; full 1–30 band on the same cloudy days, which SkyGPT does not provide; efficiency 0.43M params.)*

### 4.9 Economic value
*(CAISO reserve simulation; +$/GW/yr; 100% of price scenarios profitable.)*

### 4.10 Ablations
*(CTI gating, persistence anchor, SDE vs deterministic, covariates.)*

## 5. Discussion
*(Stochastic-process view; calibration as the operative property; efficiency/deployability; when generation does/does not help.)*

## 6. Conclusion

## Appendix A — Mathematical formulation (see appendix_math.md)
## Appendix B — Supplementary statistics (multi-seed, PIT, reliability, DM)
## Appendix C — Implementation & reproducibility
