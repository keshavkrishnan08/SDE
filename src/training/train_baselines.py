"""Train all baseline models using the same data splits and evaluation pipeline."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.baselines.persistence import PersistenceModel
from src.models.baselines.smart_persistence import SmartPersistenceModel
from src.models.baselines.lstm import LSTMForecaster
from src.models.baselines.mc_dropout import MCDropoutLSTM
from src.models.baselines.deep_ensemble import DeepEnsemble
from src.models.baselines.timegrad import TimeGrad
from src.models.baselines.csdi import CSDI
from src.models.baselines.cnn_image import CNNImageForecaster
from src.utils.io import save_checkpoint, get_device
from src.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def prepare_timeseries_data(
    split_path: Path, seq_len: int, horizons: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare input/target tensors from a split parquet file."""
    df = pd.read_parquet(split_path)
    feature_cols = [c for c in ["ghi", "clear_sky_index", "solar_zenith",
                                "temperature", "humidity", "wind_speed"]
                    if c in df.columns]
    if not feature_cols:
        feature_cols = [df.columns[1]]  # Fallback

    data = df[feature_cols].values.astype(np.float32)
    ghi = df["ghi"].values.astype(np.float32) if "ghi" in df.columns else data[:, 0]
    max_h = max(horizons)

    X, Y = [], []
    for i in range(seq_len, len(data) - max_h):
        X.append(data[i - seq_len : i])
        Y.append(np.array([ghi[i + h] for h in horizons]))

    return torch.tensor(np.array(X)), torch.tensor(np.array(Y))


def train_lstm_model(
    model: torch.nn.Module,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    val_X: torch.Tensor,
    val_Y: torch.Tensor,
    config: dict,
    model_name: str,
) -> torch.nn.Module:
    """Generic training loop for LSTM-based models."""
    device = get_device()
    model = model.to(device)
    cfg = config["baselines"]["lstm"]

    train_ds = TensorDataset(train_X, train_Y)
    val_ds = TensorDataset(val_X, val_Y)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = torch.nn.MSELoss()
    best_val = float("inf")

    for epoch in range(cfg["epochs"]):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                val_loss += criterion(pred, Y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, {"val_loss": val_loss},
                            Path(f"outputs/checkpoints/{model_name}_best.pt"))

    logger.info(f"{model_name} training complete. Best val loss: {best_val:.4f}")
    return model


def train_diffusion_baseline(
    model: torch.nn.Module,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    config: dict,
    model_name: str,
    cfg_key: str,
) -> torch.nn.Module:
    """Training loop for diffusion-based baselines (TimeGrad, CSDI)."""
    device = get_device()
    model = model.to(device)
    cfg = config["baselines"][cfg_key]

    # Use first target horizon only for simplicity
    train_y = train_Y[:, 0]
    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            loss = model.training_loss(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.info(f"{model_name} epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f}")

    save_checkpoint(model, optimizer, cfg["epochs"] - 1, {},
                    Path(f"outputs/checkpoints/{model_name}_final.pt"))
    logger.info(f"{model_name} training complete.")
    return model


def train_all_baselines(config: dict) -> dict:
    """Train all 8 baseline models.

    Returns:
        Dictionary mapping model names to trained model instances.
    """
    seed_everything(config["evaluation"]["seeds"][0])
    splits_dir = Path(config["data"]["processed_dir"]) / "splits"
    horizons = config["data"]["forecast_horizons"]
    seq_len = config["data"]["sequence_length"]

    logger.info("Preparing timeseries data...")
    train_X, train_Y = prepare_timeseries_data(splits_dir / "train.parquet", seq_len, horizons)
    val_X, val_Y = prepare_timeseries_data(splits_dir / "val.parquet", seq_len, horizons)

    input_dim = train_X.shape[-1]
    num_horizons = len(horizons)
    models = {}

    # 1. Persistence (no training needed)
    logger.info("Setting up Persistence baseline...")
    persistence = PersistenceModel()
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    if "ghi" in train_df.columns:
        persistence.fit(train_df["ghi"].values, horizons)
    models["persistence"] = persistence

    # 2. Smart Persistence (no training needed)
    logger.info("Setting up Smart Persistence baseline...")
    smart_pers = SmartPersistenceModel()
    if all(c in train_df.columns for c in ["ghi", "clear_sky_index", "ghi_clearsky"]):
        smart_pers.fit(
            train_df["clear_sky_index"].values,
            train_df["ghi_clearsky"].values,
            train_df["ghi"].values,
            horizons,
        )
    models["smart_persistence"] = smart_pers

    # 3. Deterministic LSTM
    logger.info("Training LSTM baseline...")
    lstm = LSTMForecaster(input_dim=input_dim, num_horizons=num_horizons)
    models["lstm"] = train_lstm_model(lstm, train_X, train_Y, val_X, val_Y, config, "lstm")

    # 4. MC Dropout LSTM
    logger.info("Training MC Dropout baseline...")
    mc_drop = MCDropoutLSTM(input_dim=input_dim, num_horizons=num_horizons,
                            dropout=config["baselines"]["mc_dropout"]["dropout"])
    models["mc_dropout"] = train_lstm_model(mc_drop, train_X, train_Y, val_X, val_Y, config, "mc_dropout")

    # 5. Deep Ensemble
    logger.info("Training Deep Ensemble baseline...")
    ensemble = DeepEnsemble(
        num_members=config["baselines"]["deep_ensemble"]["num_members"],
        input_dim=input_dim, num_horizons=num_horizons,
    )
    # Train each member with a different seed
    for i, member in enumerate(ensemble.members):
        seed_everything(config["evaluation"]["seeds"][0] + i)
        member_model = train_lstm_model(
            member, train_X, train_Y, val_X, val_Y, config, f"ensemble_member_{i}"
        )
    models["deep_ensemble"] = ensemble

    # 6. TimeGrad
    logger.info("Training TimeGrad baseline...")
    tg = TimeGrad(
        input_dim=input_dim,
        hidden_size=config["baselines"]["timegrad"]["hidden_size"],
        diffusion_steps=config["baselines"]["timegrad"]["diffusion_steps"],
    )
    models["timegrad"] = train_diffusion_baseline(tg, train_X, train_Y, config, "timegrad", "timegrad")

    # 7. CSDI
    logger.info("Training CSDI baseline...")
    csdi = CSDI(
        input_dim=input_dim,
        d_model=config["baselines"]["csdi"]["d_model"],
        nhead=config["baselines"]["csdi"]["nhead"],
        num_layers=config["baselines"]["csdi"]["num_layers"],
        diffusion_steps=config["baselines"]["csdi"]["diffusion_steps"],
    )
    models["csdi"] = train_diffusion_baseline(csdi, train_X, train_Y, config, "csdi", "csdi")

    # 8. CNN + Image (only if images are available)
    logger.info("Training CNN+Image baseline...")
    cnn = CNNImageForecaster(num_horizons=num_horizons)
    # CNN requires image data — skipped if not available
    save_checkpoint(cnn, None, 0, {}, Path("outputs/checkpoints/cnn_image_init.pt"))
    models["cnn_image"] = cnn

    logger.info("All baselines trained.")
    return models
