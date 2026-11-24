"""Train a configurable binary CNN, optionally under torchrun/DDP."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler

from mod_model import build_model


def read_array(spec: Any) -> np.ndarray:
    """Read .npy, .npz, or NetCDF arrays; mappings may specify a variable key."""
    if isinstance(spec, str):
        path, key = spec, None
    elif isinstance(spec, dict):
        path, key = spec["path"], spec.get("key")
    else:
        raise TypeError(f"Array spec must be a path or mapping with path/key, got {spec!r}")
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf", ".netcdf"}:
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError("Reading NetCDF requires xarray and a NetCDF backend; install with `pip install xarray netCDF4`") from exc
        with xr.open_dataset(path) as dataset:
            if key is None:
                variables = list(dataset.data_vars)
                if len(variables) != 1:
                    raise ValueError(f"{path} contains variables {variables}; provide a key")
                key = variables[0]
            if key not in dataset.data_vars:
                raise KeyError(f"Variable {key!r} not found in {path}; available: {list(dataset.data_vars)}")
            return np.asarray(dataset[key].load().values)
    loaded = np.load(path, mmap_mode="r" if suffix == ".npy" else None, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if key is None:
            if len(loaded.files) != 1:
                loaded.close()
                raise ValueError(f"{path} contains multiple arrays; provide a key")
            key = loaded.files[0]
        if key not in loaded:
            loaded.close()
            raise KeyError(f"Array key {key!r} not found in {path}")
        result = loaded[key]
        loaded.close()
        return result
    return loaded


def read_labels(spec: Any) -> np.ndarray:
    """Read a one-column CSV label file (optionally with a header), or an array file."""
    path = spec["path"] if isinstance(spec, dict) else spec
    if Path(path).suffix.lower() != ".csv":
        return read_array(spec)
    values: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.reader(stream), start=1):
            if not row or not row[0].strip():
                continue
            if len(row) != 1:
                raise ValueError(f"Label CSV {path} must have one column; row {row_number} has {len(row)}")
            try:
                values.append(float(row[0]))
            except ValueError:
                if values:
                    raise ValueError(f"Non-numeric label in {path} at row {row_number}")
                # Allow a single header row such as "label".
                continue
    return np.asarray(values, dtype=np.float32)


def make_dataset(x_spec: Any, y_spec: Any, expected: dict[str, int] | None = None) -> TensorDataset:
    x = np.asarray(read_array(x_spec), dtype=np.float32)
    y = np.asarray(read_labels(y_spec), dtype=np.float32).reshape(-1)
    if x.ndim != 4:
        raise ValueError(f"X must have shape [nb,nx,ny,nc], got {x.shape}")
    if len(y) != x.shape[0]:
        raise ValueError(f"X/Y sample counts differ: {x.shape[0]} vs {len(y)}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X and Y must contain only finite values")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Y must contain binary labels (0 or 1)")
    if expected:
        for axis, name in ((1, "nx"), (2, "ny"), (3, "nc")):
            if name in expected and x.shape[axis] != int(expected[name]):
                raise ValueError(f"Configured {name}={expected[name]} but {x.shape} has {x.shape[axis]}")
    return TensorDataset(torch.from_numpy(np.array(x, copy=True)), torch.from_numpy(y.copy()))


def split_train_validation(dataset: TensorDataset, fraction: float, seed: int):
    if not 0.0 < fraction < 1.0:
        raise ValueError("data.validation_fraction must be between 0 and 1")
    n_val = max(1, int(round(len(dataset) * fraction)))
    if n_val >= len(dataset):
        raise ValueError("Need at least two samples to create train/validation splits")
    return random_split(dataset, [len(dataset) - n_val, n_val],
                        generator=torch.Generator().manual_seed(seed))


def setup_distributed() -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    rank = dist.get_rank() if distributed else 0
    return distributed, rank, local_rank, device


def reduce_sums(values: list[float], device: torch.device, distributed: bool) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, distributed: bool) -> dict[str, float]:
    model.eval()
    loss_sum = correct = tp = fp = fn = count = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.numel()
            pred = torch.sigmoid(logits) >= 0.5
            truth = y >= 0.5
            correct += (pred == truth).sum().item()
            tp += (pred & truth).sum().item()
            fp += (pred & ~truth).sum().item()
            fn += (~pred & truth).sum().item()
            count += y.numel()
    loss_sum, correct, tp, fp, fn, count = reduce_sums(
        [loss_sum, correct, tp, fp, fn, count], device, distributed)
    return {"loss": loss_sum / max(count, 1), "accuracy": correct / max(count, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tornado.yaml", help="Path to YAML configuration")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    distributed, rank, local_rank, device = setup_distributed()
    seed = int(cfg.get("seed", 1234))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    shape = cfg.get("data", {}).get("input_shape", {})
    full_train_set = make_dataset(cfg["data"]["train_x"], cfg["data"]["train_y"], shape)
    has_val_x, has_val_y = "val_x" in cfg["data"], "val_y" in cfg["data"]
    if has_val_x != has_val_y:
        raise ValueError("Configure both data.val_x and data.val_y, or omit both to split the training pair")
    if has_val_x:
        train_set = full_train_set
        val_set = make_dataset(cfg["data"]["val_x"], cfg["data"]["val_y"], shape)
    else:
        train_set, val_set = split_train_validation(
            full_train_set, float(cfg["data"].get("validation_fraction", 0.2)), seed)
    batch_size = int(cfg["training"].get("batch_size", 32))
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=seed) if distributed else None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=int(cfg["training"].get("num_workers", 0)),
                              pin_memory=device.type == "cuda", drop_last=False)
    # DistributedSampler pads validation samples across ranks; use rank 0 evaluation
    # to keep metric counts exact, then broadcast the result to all ranks.
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=int(cfg["training"].get("num_workers", 0)),
                            pin_memory=device.type == "cuda")

    model = build_model(cfg["model"]).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_cfg["learning_rate"]),
                                  weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)))
    scheduler = None
    if optimizer_cfg.get("scheduler", {}).get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(optimizer_cfg["scheduler"].get("factor", 0.5)),
            patience=int(optimizer_cfg["scheduler"].get("patience", 3)))

    output_path = Path(cfg["training"].get("output_path", "checkpoints/tornado_cnn.pt"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs = int(optimizer_cfg["epochs"])
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = train_count = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.numel()
            train_count += y.numel()
        train_loss, train_count = reduce_sums([train_loss, train_count], device, distributed)
        # Evaluate unwrapped model on rank 0; this avoids duplicated validation data.
        if rank == 0:
            val_metrics = evaluate(model.module if distributed else model, val_loader, criterion, device, False)
            if scheduler is not None:
                scheduler.step(val_metrics["loss"])
            print(f"epoch {epoch + 1}/{epochs} train_loss={train_loss / max(train_count, 1):.5f} "
                  f"val_loss={val_metrics['loss']:.5f} val_f1={val_metrics['f1']:.4f}", flush=True)
            save_best = bool(cfg["training"].get("save_best", True))
            should_save = (val_metrics["loss"] < best_val) if save_best else True
            if should_save:
                best_val = val_metrics["loss"]
                raw_model = model.module if distributed else model
                torch.save({"model_state_dict": raw_model.state_dict(), "model": cfg["model"],
                            "input_shape": shape, "epoch": epoch + 1, "val_metrics": val_metrics,
                            "threshold": float(cfg.get("inference", {}).get("threshold", 0.5))}, output_path)
                print(f"saved best checkpoint: {output_path}", flush=True)
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
