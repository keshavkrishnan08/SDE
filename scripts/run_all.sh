#!/bin/bash
# SolarSDE: Full pipeline — download → train → evaluate → figures
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "  SolarSDE Full Pipeline"
echo "============================================"

# Step 1: Download data
echo "[1/10] Downloading datasets..."
python -m src.data.download

# Step 2: Preprocess data
echo "[2/10] Preprocessing data..."
python -c "
from src.utils.config import load_config
from src.data.preprocess import run_full_preprocessing
config = load_config('configs/default.yaml')
run_full_preprocessing(config)
"

# Step 3: Train CS-VAE (Stage 1)
echo "[3/10] Training CS-VAE..."
python -c "
from src.utils.config import load_config
from src.training.train_vae import train_vae
config = load_config('configs/default.yaml')
train_vae(config)
"

# Step 4: Extract latents + CTI (Stage 2)
echo "[4/10] Extracting latents and computing CTI..."
python -c "
from src.utils.config import load_config
from src.training.extract_latents import extract_latents
config = load_config('configs/default.yaml')
extract_latents(config)
"

# Step 5: Train Neural SDE (Stage 3)
echo "[5/10] Training Latent Neural SDE..."
python -c "
from src.utils.config import load_config
from src.training.train_sde import train_sde
config = load_config('configs/default.yaml')
train_sde(config)
"

# Step 6: Train Score Decoder (Stage 4)
echo "[6/10] Training Score-Matching Decoder..."
python -c "
from src.utils.config import load_config
from src.training.train_score import train_score_decoder
config = load_config('configs/default.yaml')
train_score_decoder(config)
"

# Step 7: Optional end-to-end fine-tuning (Stage 5)
echo "[7/10] Fine-tuning end-to-end..."
python -c "
from src.utils.config import load_config
from src.training.finetune import finetune
config = load_config('configs/default.yaml')
finetune(config)
"

# Step 8: Train all baselines
echo "[8/10] Training baselines..."
python -c "
from src.utils.config import load_config
from src.training.train_baselines import train_all_baselines
config = load_config('configs/default.yaml')
train_all_baselines(config)
"

# Step 9: Evaluate all models
echo "[9/10] Running evaluation..."
echo "  (Main experiment, ablations, CTI analysis, economic value, etc.)"
echo "  See individual evaluation scripts for details."

# Step 10: Generate figures
echo "[10/10] Generating figures..."
python -c "
from src.visualization.architecture_diagram import plot_architecture_diagram
plot_architecture_diagram()
print('Architecture diagram generated.')
"

echo "============================================"
echo "  Pipeline complete!"
echo "  Checkpoints: outputs/checkpoints/"
echo "  Figures: outputs/figures/"
echo "  Results: outputs/results/"
echo "============================================"
