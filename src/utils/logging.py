"""Experiment logging utilities (W&B or TensorBoard fallback)."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Unified logging interface supporting W&B and local CSV fallback."""

    def __init__(
        self,
        project: str = "solar-sde",
        run_name: str | None = None,
        use_wandb: bool = False,
        log_dir: str | Path = "outputs/logs",
    ):
        self.use_wandb = use_wandb
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._run = None
        self._csv_path = self.log_dir / f"{run_name or 'run'}.csv"
        self._csv_header_written = False

        if use_wandb:
            try:
                import wandb
                self._run = wandb.init(project=project, name=run_name)
            except ImportError:
                logger.warning("wandb not installed, falling back to CSV logging")
                self.use_wandb = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of metrics."""
        if self.use_wandb and self._run is not None:
            import wandb
            wandb.log(metrics, step=step)

        # Always log to CSV as backup
        import csv
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + sorted(metrics.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            row = {"step": step}
            row.update(metrics)
            writer.writerow(row)

    def finish(self) -> None:
        """Finalize the logging run."""
        if self.use_wandb and self._run is not None:
            import wandb
            wandb.finish()
