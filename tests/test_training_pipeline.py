import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml


ROOT = Path(__file__).resolve().parents[1]


def write_pair(directory: Path, stem: str, n: int, seed: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8, 8, 1)).astype("float32")
    y = np.arange(n, dtype=np.int64) % 2
    x_path = directory / f"{stem}_x.nc"
    y_path = directory / f"{stem}_y.csv"
    xr.Dataset({"fields": (("sample", "x", "y", "channel"), x)}).to_netcdf(x_path)
    with y_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label"])
        writer.writerows([[int(label)] for label in y])
    return x_path, y_path


def test_netcdf_csv_training_and_checkpoint(tmp_path: Path) -> None:
    train_x, train_y = write_pair(tmp_path, "train", 10, 1)
    test_x, test_y = write_pair(tmp_path, "test", 4, 2)
    checkpoint = tmp_path / "checkpoint.pt"
    config = {
        "seed": 11,
        "data": {
            "input_shape": {"nx": 8, "ny": 8, "nc": 1},
            "train_x": str(train_x),
            "train_y": str(train_y),
            "validation_fraction": 0.2,
        },
        "model": {"name": "TornadoCNN", "params": {
            "input_channels": 1, "channels": [4], "kernel_size": 3,
            "pool_size": 2, "dropout": 0.0, "fc_nodes": [4],
        }},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001,
                     "weight_decay": 0.0, "num_workers": 0, "save_best": True,
                     "output_path": str(checkpoint)},
        "inference": {"threshold": 0.5, "test_x": str(test_x), "test_y": str(test_y)},
    }
    config_path = tmp_path / "ci.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run([sys.executable, str(ROOT / "scripts/main.py"), "--config", str(config_path)],
                   check=True, cwd=ROOT, timeout=120)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["epoch"] == 1
    assert "model_state_dict" in state
