#!/bin/bash
# Train all baseline models
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Training All Baselines"
echo "============================================"

python -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.utils.config import load_config
from src.training.train_baselines import train_all_baselines
config = load_config('configs/default.yaml')
train_all_baselines(config)
"

echo "============================================"
echo "  All baselines trained!"
echo "============================================"
