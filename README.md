# Tornado dynamics CNN classifier

This repository includes a configurable PyTorch CNN for binary classification of gridded tornado input fields. It accepts feature batches shaped `[nb, nx, ny, nc]` and labels shaped `[nb]` with values 0 or 1. The network returns one logit per sample; inference converts logits to probabilities with sigmoid.

## Files

- [`scripts/mod_model.py`](scripts/mod_model.py): `TornadoCNN`, configurable architecture, and model registry/factory for swapping architectures.
- [`scripts/tornado.yaml`](scripts/tornado.yaml): data paths, input dimensions, architecture, optimizer, training, checkpoint, and inference settings.
- [`scripts/main.py`](scripts/main.py): training loop with optional PyTorch DistributedDataParallel (DDP).
- [`scripts/inference.ipynb`](scripts/inference.ipynb): loads the same YAML and trained checkpoint, predicts test probabilities, computes precision/recall/F1, plots metrics and a confusion matrix, and saves probabilities.

## Install

Use Python 3.10 or newer. Install a PyTorch build appropriate for your CPU/GPU system, then install the other dependencies:

```bash
python -m pip install numpy pyyaml matplotlib scikit-learn jupyter xarray netCDF4
```

## Prepare data

Training features are read from `train_x.nc`, a NetCDF file containing one array shaped `[nb, nx, ny, nc]`. Labels are read from `train_y.csv`, a one-column CSV with `nb` values of 0 or 1; an optional header such as `label` is accepted. Test inference uses the same pairing, configured as `inference.test_x` and `inference.test_y`. NetCDF (`.nc`, `.nc4`, `.cdf`, `.netcdf`) is read through xarray and the `netCDF4` backend. The sample YAML uses:

```yaml
data:
  train_x: data/train_x.nc
  train_y: data/train_y.csv
```

If separate validation files are unavailable, omit `val_x` and `val_y`; the training script reserves `data.validation_fraction` (default sample: 0.2) from the training pair for validation. You can instead provide both `val_x` (NetCDF) and `val_y` (CSV) to use a separate validation set. `.npy` and `.npz` inputs remain supported for feature arrays, and array labels are also supported.

Set `data.input_shape.nx`, `ny`, and `nc` in the YAML to match each sample. Make sure `model.params.input_channels` equals `nc`. The sample YAML points to placeholder files under `data/`; edit the paths for your data.

## Train on one device

From the repository root:

```bash
python scripts/main.py --config scripts/tornado.yaml
```

Or from `scripts/`:

```bash
cd scripts
python main.py --config tornado.yaml
```

Paths in the YAML are resolved from the current working directory. The default checkpoint is `checkpoints/tornado_cnn.pt` relative to that directory; create/edit `training.output_path` as desired. The model with the lowest validation BCE loss is saved when `training.save_best: true`. With it set to `false`, the checkpoint is overwritten each epoch with the latest model.

## Train with DDP / torchrun

On one machine with 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 scripts/main.py --config scripts/tornado.yaml
```

From inside `scripts/`, use:

```bash
torchrun --standalone --nproc_per_node=4 main.py --config tornado.yaml
```

For multi-node jobs, launch `torchrun` on each node with the same rendezvous settings (`--nnodes`, `--node_rank`, `--master_addr`, and `--master_port`) and accessible shared data/checkpoint paths. The code selects NCCL with CUDA and Gloo on CPU. In DDP mode each rank receives a data shard; only rank 0 writes checkpoints and prints validation metrics. Choose a per-rank `training.batch_size` that fits device memory.

## Configure the model

`model.name` selects an architecture in `MODEL_REGISTRY` in `mod_model.py`. `TornadoCNN` accepts `input_channels`, a list of convolution output `channels` (list length controls convolution depth), `kernel_size`, `pool_size`, `dropout`, and `fc_nodes` (dense head widths). Its adaptive global pooling supports different spatial dimensions. To replace it, implement an `nn.Module` that accepts `[B,nx,ny,nc]` or `[B,nc,nx,ny]` and returns a `[B]` logit tensor, then register it in `MODEL_REGISTRY` and set its name/parameters in YAML.

The YAML also controls epochs, batch size, learning rate, weight decay, workers, optional ReduceLROnPlateau scheduling, save-best behavior, output path, and the probability threshold used by inference.

## Run inference and evaluation

After training, start Jupyter from the repository root:

```bash
jupyter lab scripts/inference.ipynb
```

Run all cells. The notebook reads `scripts/tornado.yaml`, including `inference.test_x`, `inference.test_y`, and `training.output_path`. Update those paths first. It reports test precision, recall, and F1 at the configured threshold, plots the three scores and confusion matrix, and writes `test_probabilities.npy` in the notebook's working directory. Change `inference.threshold` to tune the classification cutoff; probabilities themselves are independent of the threshold.

## Continuous integration

GitHub Actions runs on every push to `main` and on pull requests. The workflow installs CPU dependencies and runs an end-to-end smoke test that creates a small NetCDF feature file and CSV label file, trains for one epoch, and checks that a checkpoint is written. The workflow is defined in `.github/workflows/ci.yml`.

## Notes

- Checkpoints store model weights, model configuration, input dimensions, epoch, validation metrics, and inference threshold.
- The training script validates input rank, dimensions, finite values, label count, and binary labels.
- The example does not normalize features automatically. Apply a consistent preprocessing/normalization step to train, validation, and test fields before saving them if needed.
- For very large arrays, the current implementation loads array data into memory for each launched process. Ensure each rank/node has enough memory.
