"""
Convert one DSEC-Det sequence into the flat format expected by frame_construction/main_dsec.py:
  output_dir/
    train/
      {seq}_bbox.npy   <- Prophesee-format labels (tracks.npy)
      {seq}.h5         <- combined events + frames H5
    test/
      ...

The H5 format expected by FAOD main_dsec.py:
  events/t, events/x, events/y, events/p
  frames/image        (N, H, W, C) uint8
  frames/timestamp    (N,) int64 microseconds
"""
import os, cv2
import numpy as np
import h5py
from pathlib import Path

import importlib.util
_spec = importlib.util.find_spec('hdf5plugin')
if _spec:
    os.environ['HDF5_PLUGIN_PATH'] = str(Path(_spec.origin).parent / 'plugins')
import hdf5plugin  # noqa

DSEC_ROOT = Path('/home/karthik/Desktop/spiking_neural_networks/datasets/dsec_data')
OUT_ROOT  = Path('/home/karthik/Desktop/spiking_neural_networks/FAOD-master/data/dsec_small_flat')

SEQUENCES = {
    'train': ['interlaken_00_c'],
    'test':  ['interlaken_00_a'],
}


def load_image_timestamps(seq_path: Path) -> np.ndarray:
    ts_file = seq_path / 'images/timestamps.txt'
    if ts_file.exists():
        return np.genfromtxt(ts_file, dtype='int64')
    # Fall back: derive from events h5 ms_to_idx
    events_file = seq_path / 'events/left/events.h5'
    img_files = sorted((seq_path / 'images/left/distorted').glob('*.png'))
    n = len(img_files)
    with h5py.File(events_file, 'r') as f:
        t_offset = int(f['t_offset'][()] if 't_offset' in f else 0)
        ms_to_idx = f['ms_to_idx'][:]
    total_ms = len(ms_to_idx)
    step = total_ms // max(n, 1)
    ts = np.array([ms_to_idx[min(i * step, total_ms - 1)] for i in range(n)], dtype='int64')
    return ts + t_offset


def convert_sequence(dsec_seq_path: Path, out_split_dir: Path, seq_name: str):
    out_split_dir.mkdir(parents=True, exist_ok=True)
    dst_h5  = out_split_dir / f'{seq_name}.h5'
    dst_npy = out_split_dir / f'{seq_name}_bbox.npy'

    # --- Labels ---
    src_npy = dsec_seq_path / 'object_detections/left/tracks.npy'
    if not dst_npy.exists():
        tracks = np.load(src_npy)
        np.save(dst_npy, tracks)
        print(f'  Labels: {len(tracks):,} detections -> {dst_npy.name}')
    else:
        print(f'  Labels already exist')

    if dst_h5.exists():
        print(f'  H5 already exists')
        return

    # --- Load events ---
    src_h5 = dsec_seq_path / 'events/left/events.h5'
    print(f'  Loading events...')
    with h5py.File(src_h5, 'r') as fin:
        t_offset = int(fin['t_offset'][()] if 't_offset' in fin else 0)
        ev_t = fin['events/t'][:].astype(np.int64) + t_offset
        ev_x = fin['events/x'][:]
        ev_y = fin['events/y'][:]
        ev_p = fin['events/p'][:]
    print(f'  Events: {len(ev_t):,}')

    # --- Load images ---
    img_dir = dsec_seq_path / 'images/left/distorted'
    img_files = sorted(img_dir.glob('*.png'))
    print(f'  Loading {len(img_files)} images...')
    img_ts = load_image_timestamps(dsec_seq_path)

    # Align: only keep images that have timestamps
    n_imgs = min(len(img_files), len(img_ts))
    img_files = img_files[:n_imgs]
    img_ts    = img_ts[:n_imgs]

    sample_img = cv2.imread(str(img_files[0]))
    H, W, C = sample_img.shape
    images = np.zeros((n_imgs, H, W, C), dtype=np.uint8)
    for i, f in enumerate(img_files):
        img = cv2.imread(str(f))
        if img is not None:
            images[i] = img

    # --- Write combined H5 ---
    print(f'  Writing H5: events + {n_imgs} frames...')
    with h5py.File(dst_h5, 'w') as fout:
        ev_grp = fout.create_group('events')
        ev_grp.create_dataset('t', data=ev_t, dtype=np.int64)
        ev_grp.create_dataset('x', data=ev_x, dtype=np.int16)
        ev_grp.create_dataset('y', data=ev_y, dtype=np.int16)
        ev_grp.create_dataset('p', data=ev_p.astype(np.int8), dtype=np.int8)

        fr_grp = fout.create_group('frames')
        fr_grp.create_dataset('image',     data=images,  dtype=np.uint8,  chunks=(1, H, W, C))
        fr_grp.create_dataset('timestamp', data=img_ts,  dtype=np.int64)
    print(f'  Done -> {dst_h5.name}')


if __name__ == '__main__':
    print('=== Preparing small DSEC dataset for FAOD ===')
    print(f'DSEC_ROOT: {DSEC_ROOT}')
    print(f'OUT_ROOT:  {OUT_ROOT}\n')

    for split, seqs in SEQUENCES.items():
        for seq in seqs:
            dsec_path = DSEC_ROOT / split / seq
            if not dsec_path.exists():
                print(f'[SKIP] {split}/{seq} not found'); continue
            print(f'[{split}] {seq}')
            convert_sequence(dsec_path, OUT_ROOT / split, seq)
            print()

    print('Done. Now run:')
    print(f'  python3 frame_construction/main_dsec.py \\')
    print(f'    --input_dir {OUT_ROOT} \\')
    print(f'    --target_dir data/dsec_small_h5 \\')
    print(f'    --num_processes 2')
