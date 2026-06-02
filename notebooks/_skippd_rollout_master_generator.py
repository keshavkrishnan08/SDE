"""Build notebook 10: SolarSDE-ROLLOUT on SKIPP'D.

Identical experiment suite to notebook 09, but the SDE marginal is produced by
ROLLING THE LATENT FORWARD (Euler-Maruyama) and decoding the future latent —
the latent-rollout / cloud-state-evolution variant (original CLAUDE.md design),
the method-level attempt to match SkyGPT's cloud-motion modeling.

Reuses 09's whole pipeline + downstream verbatim (the rollout model is a drop-in
with the same (pi, mean, std) interface, aliased to TemporalLatentSDE). Only two
cells differ: the architecture (ROLLOUT_ARCH_CODE) and the ablation
(ABLATIONS_ROLLOUT_CODE). Separate output dir (skippd_rollout_run) so it trains
its own model rather than reusing 09's checkpoint.
"""
import json, sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR))

# Reuse all of 09's building blocks
from _skippd_master_generator import (
    SKIPPD_SETUP_CODE, SKIPPD_DOWNLOAD_FULL_CODE, SKIPPD_PREP_CODE, SKIPPD_VAE_CODE,
    SKIPPD_LATENTS_WRITE_CODE, SKIPPD_HORIZON_OVERRIDE_CODE,
    SHARED_CODE, LOAD_DATA_TOLERANT_CODE, BASELINES_CODE, STRATIFIED_CODE, ANALYSIS_CODE,
    RAMP_AUROC_CODE, BOOTSTRAP_CIS_CODE, PIT_RELIABILITY_CODE, ECONOMIC_CAISO_CODE,
    LATEX_TABLES_CODE, ZIP_DOWNLOAD_CODE, CTI_VALIDATION_CODE, HOLM_BONFERRONI_CODE,
    safe_stage, build_nb,
    STAGE_0_V2_CODE, POST_STAGE0_V2_VERIFY_CODE,
    IMPLEMENTATION_DETAILS_CODE, DATA_CARD_CODE, COMPUTATIONAL_COST_CODE,
    RELIABILITY_LEVELS_CODE, SAMPLING_EFFICIENCY_CODE, ECONOMIC_SENSITIVITY_CODE,
    CROSS_VALIDATION_V2_CODE, SKYGPT_BENCHMARK_CODE,
)
# The two cells that differ for the rollout architecture
from _solarsde_rollout import ROLLOUT_ARCH_CODE, ABLATIONS_ROLLOUT_CODE

# Separate output dir so notebook 10 trains its own rollout model (doesn't reuse 09's ckpt)
ROLLOUT_SETUP_CODE = SKIPPD_SETUP_CODE.replace("skippd_run", "skippd_rollout_run")


def nb_skippd_rollout():
    cells = [
        ("markdown",
         "# SolarSDE-ROLLOUT on SKIPP'D — Latent-Rollout PV Nowcasting\n\n"
         "Same 497-day SKIPP'D benchmark and same experiment suite as notebook 09, but the "
         "forecast is produced by **rolling the cloud-state latent forward** with a learned "
         "neural SDE (Euler-Maruyama, n_paths sample paths) and **decoding the future latent** "
         "to a PV distribution — modeling cloud-state *evolution* (advection) rather than reading "
         "a closed-form marginal. This is the method-level attempt to match SkyGPT's cloud-motion "
         "modeling, in latent space (no pixel generation). Run on a **GPU** runtime; training is "
         "slower than 09 (the rollout backprops through up to 30 Euler steps)."),

        ("markdown", "## 0. Setup"),
        ("code", ROLLOUT_SETUP_CODE),
        ("markdown", "## 1. Download full SKIPP'D (~2.3 GB)"),
        ("code", SKIPPD_DOWNLOAD_FULL_CODE),
        ("markdown", "## 2. Preprocess — clear-sky-PV index, ramps, chronological splits"),
        ("code", SKIPPD_PREP_CODE),
        ("markdown", "## 3. Train CS-VAE (64×64 → 64-d latent) + encode + optical-flow motion"),
        ("code", "Z_DIM = 64\nSKIPPD_VAE_EPOCHS = 12\n" + SKIPPD_VAE_CODE),
        ("markdown", "## 4. CTI + write the {splits, extended, latents} contract"),
        ("code", SKIPPD_LATENTS_WRITE_CODE),

        ("markdown", "## 5. Shared metrics + load tensors (CTI normalized here)"),
        ("code", SHARED_CODE),
        ("code", LOAD_DATA_TOLERANT_CODE),
        ("code", SKIPPD_HORIZON_OVERRIDE_CODE),
        ("markdown", "## 5a. Data card + implementation details"),
        ("code", safe_stage("DATA_CARD", DATA_CARD_CODE)),
        ("code", safe_stage("IMPLEMENTATION_DETAILS", IMPLEMENTATION_DETAILS_CODE)),

        ("markdown", "## 6. Train ROLLOUT Latent Neural SDE (Euler-Maruyama rollout + decoder + persistence-blend + Mondrian calibration)"),
        ("code", ROLLOUT_ARCH_CODE),
        ("code", STAGE_0_V2_CODE),
        ("code", safe_stage("POST_STAGE0_V2_VERIFY", POST_STAGE0_V2_VERIFY_CODE)),

        ("markdown", "## 6b. SkyGPT exact-benchmark — identical Nov-Dec 2019 cloudy test (full 1–30 band)"),
        ("code", safe_stage("SKYGPT_BENCHMARK", SKYGPT_BENCHMARK_CODE)),

        ("markdown", "## 7. Baselines (persistence, smart-persistence, LSTM, MC-Dropout, CSDI)"),
        ("code", safe_stage("BASELINES", BASELINES_CODE)),
        ("markdown", "## 8. Ablations (rollout-native: A2 no-CTI, A4 no-persistence, A5 no-diffusion/ODE, A7 no-cov)"),
        ("code", safe_stage("ABLATIONS_ROLLOUT", ABLATIONS_ROLLOUT_CODE)),

        ("markdown", "## 9. Stratified eval + Diebold-Mariano significance"),
        ("code", safe_stage("STRATIFIED", STRATIFIED_CODE)),
        ("markdown", "## 9a. Leave-one-month-out cross-validation"),
        ("code", safe_stage("CROSS_VALIDATION_V2", CROSS_VALIDATION_V2_CODE)),

        ("markdown", "## 10. PIT / reliability + bootstrap CIs"),
        ("code", safe_stage("PIT_RELIABILITY", PIT_RELIABILITY_CODE)),
        ("code", safe_stage("BOOTSTRAP_CIS", BOOTSTRAP_CIS_CODE)),
        ("markdown", "## 11. Ramp AUROC + CTI validation"),
        ("code", safe_stage("RAMP_AUROC", RAMP_AUROC_CODE)),
        ("code", safe_stage("CTI_VALIDATION", CTI_VALIDATION_CODE)),

        ("markdown", "## 12. Reliability levels + sampling efficiency + compute cost"),
        ("code", safe_stage("RELIABILITY_LEVELS", RELIABILITY_LEVELS_CODE)),
        ("code", safe_stage("SAMPLING_EFFICIENCY", SAMPLING_EFFICIENCY_CODE)),
        ("code", safe_stage("COMPUTATIONAL_COST", COMPUTATIONAL_COST_CODE)),

        ("markdown", "## 13. Economic value + sensitivity + Holm-Bonferroni"),
        ("code", safe_stage("HOLM_BONFERRONI", HOLM_BONFERRONI_CODE)),
        ("code", safe_stage("ECONOMIC_CAISO", ECONOMIC_CAISO_CODE)),
        ("code", safe_stage("ECONOMIC_SENSITIVITY", ECONOMIC_SENSITIVITY_CODE)),
        ("markdown", "## 14. Analysis figures + LaTeX tables"),
        ("code", safe_stage("ANALYSIS", ANALYSIS_CODE)),
        ("code", safe_stage("LATEX_TABLES", LATEX_TABLES_CODE)),
        ("markdown", "## Final — Zip the paper package"),
        ("code", ZIP_DOWNLOAD_CODE),
    ]
    return build_nb(cells)


if __name__ == "__main__":
    path = NB_DIR / "10_skippd_rollout_master.ipynb"
    nb = nb_skippd_rollout()
    path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {path.name}: {path.stat().st_size / 1024:.1f} KB ({len(nb['cells'])} cells)")
