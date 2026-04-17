#!/bin/bash
# Run all 7 ablation experiments
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SolarSDE Ablation Study"
echo "============================================"

ABLATIONS=(
    "ablation_no_cti:A2 — SolarSDE without CTI gating"
    "ablation_no_vae:A3 — SolarSDE without CS-VAE (raw pixels)"
    "ablation_no_score:A4 — SolarSDE without Score Matching"
    "ablation_ode:A5 — Deterministic ODE (no stochastic)"
    "ablation_adjoint:A6 — Adjoint training instead of SDE Matching"
    "ablation_no_meteo:A7 — No meteorological covariates"
)

for entry in "${ABLATIONS[@]}"; do
    config_name="${entry%%:*}"
    description="${entry##*:}"
    echo ""
    echo "--- Running ${description} ---"
    echo "Config: configs/${config_name}.yaml"

    python -c "
from src.utils.config import load_config_with_overrides
from src.training.train_sde import train_sde
config = load_config_with_overrides('configs/default.yaml', 'configs/${config_name}.yaml')
train_sde(config)
" || echo "WARNING: ${config_name} failed, continuing..."

done

echo ""
echo "============================================"
echo "  Ablation study complete!"
echo "============================================"
