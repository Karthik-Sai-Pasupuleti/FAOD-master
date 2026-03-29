# FAOD Training Guide — DSEC-Det on Custom Data

This guide documents how to train FAOD on DSEC-Det using a small subset,
including all bug fixes applied to the original repo.

---

## 1. Environment Setup

We use [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment and install dependencies
```bash
# Create a Python 3.11 venv
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install PyTorch (CUDA 11.8) + all dependencies in one shot
uv pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple
```

Or use `pyproject.toml` (syncs everything automatically):
```bash
uv sync
source .venv/bin/activate
```

### Add/remove packages
```bash
uv add <package>          # add and update pyproject.toml
uv remove <package>       # remove
uv pip install <package>  # ad-hoc install without touching pyproject.toml
```

### mmcv (optional — only if model.backbone.enable_align=True)
```bash
uv pip install openmim
mim install mmcv
```

> **Note:** The original repo pins `wandb==0.14.0`, which uses `np.float_`
> removed in NumPy 1.24+. `requirements.txt` uses `wandb>=0.16`.

---

## 2. Dataset Preparation

### Step 1 — Raw DSEC-Det download

Download the DSEC-Det dataset from the official source:
- Events: `events/left/events.h5` (with blosc compression)
- Images: `images/left/distorted/*.png`
- Labels: `object_detections/left/tracks.npy`

Expected structure:
```
datasets/dsec_data/
  train/
    interlaken_00_c/
      events/left/events.h5
      images/left/distorted/*.png
      images/timestamps.txt        # may be absent — handled automatically
      object_detections/left/tracks.npy
  test/
    interlaken_00_a/
      ...
```

### Step 2 — Convert to FAOD flat format

```bash
python prepare_dsec_small.py
```

This produces:
```
data/dsec_small_flat/
  train/interlaken_00_c.h5   +  interlaken_00_c_bbox.npy
  test/interlaken_00_a.h5    +  interlaken_00_a_bbox.npy
```

### Step 3 — Build stacked histogram representations

```bash
python frame_construction/main_dsec.py \
    --input_dir data/dsec_small_flat \
    --target_dir data/dsec_small_h5 \
    --num_processes 2
```

This produces:
```
data/dsec_small_h5/freq_1_1/
  train/  val/  test/
```

> **Note:** FAOD requires a `val/` directory. If only `test/` exists, symlink it:
> ```bash
> ln -s test data/dsec_small_h5/freq_1_1/val
> ```

---

## 3. Training

### Small-scale training (single sequence, no wandb, no deformable alignment)

```bash
WANDB_MODE=disabled python train.py \
    dataset=dsec \
    dataset.path=data/dsec_small_h5/freq_1_1 \
    +experiment/dsec=tiny.yaml \
    model.backbone.enable_align=False \
    batch_size.train=2 \
    batch_size.eval=1 \
    hardware.num_workers.train=1 \
    hardware.num_workers.eval=1 \
    training.max_steps=2000 \
    training.lr_scheduler.total_steps=2000 \
    validation.val_check_interval=500 \
    dataset.train.sampling=stream
```

Key flags explained:

| Flag | Reason |
|------|--------|
| `WANDB_MODE=disabled` | Skip wandb logging (no API key needed) |
| `model.backbone.enable_align=False` | Skip mmcv DeformConv2d (CUDA version mismatch) |
| `dataset.train.sampling=stream` | Required when training with only 1 sequence |
| `hardware.num_workers.eval=1` | Required when val set has only 1 sequence |
| `training.lr_scheduler.total_steps=2000` | Must match `max_steps` to avoid ZeroDivisionError |

### Full training (original DSEC-Det, with wandb)

```bash
wandb login   # set API key once

python train.py \
    dataset=dsec \
    dataset.path=/path/to/dsec_full_h5/freq_1_1 \
    +experiment/dsec=base.yaml \
    batch_size.train=8 \
    hardware.num_workers.train=4 \
    hardware.num_workers.eval=4
```

---

## 4. Validation / Inference

```bash
python validation.py \
    dataset=dsec \
    dataset.path=data/dsec_small_h5/freq_1_1 \
    checkpoint=dummy/<run_id>/checkpoints/<checkpoint>.ckpt \
    use_test_set=True \
    +experiment/dsec=tiny.yaml \
    model.backbone.enable_align=False \
    batch_size.eval=1 \
    hardware.num_workers.eval=1
```

---

## 5. Visualization (demo.py)

```bash
python demo.py \
    dataset=dsec \
    dataset.path=data/dsec_small_h5/freq_1_1 \
    checkpoint=dummy/<run_id>/checkpoints/<checkpoint>.ckpt \
    +experiment/dsec=tiny.yaml \
    model.backbone.enable_align=False
```

Results saved to `./predictions/` or `./gt/`.

---

## 6. Bug Fixes Applied (vs. original repo)

### `loggers/wandb_logger.py`
- `_get_public_run()`: wrapped `wandb.Api()` in try/except — avoids crash when
  `WANDB_MODE=disabled` (wandb still creates a real Run but has no API key)
- `_num_logged_artifact()`: guard for `None` public run
- `_rm_but_top_k()`: guard for `None` public run

### `basicsr/models/archs/arch_util.py`
- Wrapped `from mmcv.ops import DeformConv2d` in try/except to handle CUDA
  version mismatch (mmcv built for CUDA 11, system CUDA 13)
- `DCNv2Pack.__init__`: raises informative `RuntimeError` only when actually
  instantiated with `enable_align=True`

### `models/detection/recurrent_backbone/darknet_rnn_forward_fusion.py`
- Fixed typo `img_input_aligned == None` → `img_input_aligned = img_input`
- Added `img_unaligned = None` in else branch (was unbound when `enable_align=False`)

### `wandb` version
- Original pins `wandb==0.14.0` which crashes with NumPy 1.26.3 (`np.float_` removed)
- Use `wandb>=0.16`

---

## 7. Checkpoints

Checkpoints are saved to:
```
dummy/<wandb_run_id>/checkpoints/
  epoch=NNN-step=NNNN-val_AP=X.XX.ckpt   # best model
  last_epoch=NNN-step=NNNN.ckpt           # latest
```
