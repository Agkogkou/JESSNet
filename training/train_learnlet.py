"""Train the spherical-wavelet learnlet used by JESSNet.

Trains on simulated foreground maps (e.g. Galactic free-free, synchrotron, and
point sources). Each map is normalized, decomposed into spherical-wavelet scales,
and cut into aligned multiscale patch cubes; white noise of random amplitude is
added and the network learns to reconstruct the clean normalized map patch. This
matches exactly the representation used at inference inside JESSNet.

Example
-------
python training/train_learnlet.py \
  --input_dir /path/to/fg_training \
  --components gal_ff gal_synch point_sources \
  --output weights/learnlet_sphere_64_5_sc5_fg.pth \
  --cache_dir /path/to/cache_learnlet_sc5_ncut4 \
  --nside 256 --nscales 5 --nside_cut 4 \
  --channel_start 0 --channel_end 100 \
  --sigma_max 0.5 --epochs 1000 --batch_size 8 --lr 1e-4 --patience 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

import healpy as hp

# make the `jessnet` package importable when running from the repo without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jessnet import harmonics as hpyt
from jessnet.patches import from_healpix_to_maps_new_numba
from jessnet.learnlet import Learnlet


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------
def _iter_datasets(h5obj, prefix=""):
    for key, obj in h5obj.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            yield path, obj
        elif isinstance(obj, h5py.Group):
            yield from _iter_datasets(obj, path)


def _candidate_datasets(h5_path, nside, dataset_key=None):
    npix = hp.nside2npix(nside)
    with h5py.File(h5_path, "r") as f:
        if dataset_key not in (None, "", "auto"):
            if dataset_key not in f:
                raise KeyError(f"dataset_key={dataset_key!r} not in {h5_path}; keys: {list(f.keys())}")
            dset = f[dataset_key]
            return [(dataset_key, tuple(dset.shape), str(dset.dtype))]
        candidates = []
        for name, dset in _iter_datasets(f):
            if not np.issubdtype(dset.dtype, np.number):
                continue
            shape = tuple(dset.shape)
            if len(shape) and shape[-1] == npix:
                candidates.append((name, shape, str(dset.dtype)))
        return candidates


def _load_maps_from_file(h5_path, nside, dataset_key, channel_start, channel_end,
                         max_maps_per_file, remove_mean, normalize):
    npix = hp.nside2npix(nside)
    candidates = _candidate_datasets(h5_path, nside, dataset_key)
    if not candidates:
        raise ValueError(f"No numeric dataset with last dim npix={npix} in {h5_path}.")

    preferred = ("map", "maps", "cube", "fg", "foreground", "gal", "synch", "ff", "point")
    candidates.sort(key=lambda x: (not any(w in x[0].lower() for w in preferred), -int(np.prod(x[1]))))
    key = candidates[0][0]

    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f[key])
    arr = arr.reshape(-1, npix).astype(np.float32)

    start = 0 if channel_start is None else channel_start
    end = arr.shape[0] if channel_end is None else min(channel_end, arr.shape[0])
    arr = arr[start:end]
    if max_maps_per_file is not None:
        arr = arr[:max_maps_per_file]
    if arr.size == 0:
        raise ValueError(f"No maps left after slicing {h5_path}:{key}.")

    if remove_mean:
        arr = arr - np.mean(arr, axis=1, keepdims=True)
    if normalize:
        std = np.std(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(std, 1e-12)

    print(f"[load] {h5_path} :: {key} -> {arr.shape}")
    return arr


def discover_hdf5_files(input_dir, components):
    files = []
    for comp in components:
        comp_dir = input_dir / comp
        if not comp_dir.exists():
            raise FileNotFoundError(f"Component directory not found: {comp_dir}")
        comp_files = sorted(comp_dir.glob("*.hd5")) + sorted(comp_dir.glob("*.hdf5")) + sorted(comp_dir.glob("*.h5"))
        if not comp_files:
            raise FileNotFoundError(f"No HDF5 files found in {comp_dir}")
        files.extend(comp_files)
    print("[discover] Files to use:")
    for f in files:
        print(f"  - {f}")
    return files


# ---------------------------------------------------------------------------
# Patch cache (spherical-wavelet multiscale patch cubes)
# ---------------------------------------------------------------------------
def build_wavelet_patches_for_maps(maps, nscales, nside_cut, lmax=None):
    if lmax is None:
        lmax = 3 * hp.get_nside(maps[0])
    all_x, all_y = [], []
    for i in range(maps.shape[0]):
        m = maps[i:i + 1]                                             # (1, npix)
        wts = hpyt.wt_trans(m, nscales=nscales - 1, lmax=lmax, alm_in=False)  # (1, npix, nscales)
        if wts.shape[-1] != nscales:
            raise RuntimeError(f"Expected {nscales} wavelet channels, got {wts.shape}")
        wts_flat = np.swapaxes(wts, 1, 2).reshape(nscales, -1).astype(np.float32)   # (nscales, npix)
        patches_per_scale = from_healpix_to_maps_new_numba(wts_flat, nside_cut=nside_cut)  # (nscales, np, H, W)
        x = np.moveaxis(patches_per_scale, 0, 1)                     # (n_patches, nscales, H, W)
        y = np.sum(x, axis=1, keepdims=True)                        # (n_patches, 1, H, W) = clean map patch
        all_x.append(x.astype(np.float32))
        all_y.append(y.astype(np.float32))
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)


def _cache_name_for_file(path, args, chunk_id):
    key = json.dumps({"path": str(path), "mtime": os.path.getmtime(path), "chunk_id": chunk_id,
                      "nside": args.nside, "nscales": args.nscales, "nside_cut": args.nside_cut,
                      "channel_start": args.channel_start, "channel_end": args.channel_end,
                      "max_maps_per_file": args.max_maps_per_file, "remove_mean": args.remove_mean,
                      "normalize": args.normalize, "dataset_key": args.dataset_key}, sort_keys=True).encode()
    return f"{path.stem}_chunk{chunk_id:04d}_{hashlib.md5(key).hexdigest()[:16]}.npz"


def build_or_load_cache(args):
    input_dir = Path(args.input_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    h5_files = discover_hdf5_files(input_dir, args.components)
    shard_paths = []
    for h5_path in h5_files:
        maps = _load_maps_from_file(h5_path, args.nside, args.dataset_key, args.channel_start,
                                    args.channel_end, args.max_maps_per_file, args.remove_mean, args.normalize)
        chunk_size = args.maps_per_cache_shard
        n_chunks = int(np.ceil(maps.shape[0] / chunk_size))
        for chunk_id in range(n_chunks):
            start = chunk_id * chunk_size
            stop = min((chunk_id + 1) * chunk_size, maps.shape[0])
            out_path = cache_dir / _cache_name_for_file(h5_path, args, chunk_id)
            shard_paths.append(out_path)
            if out_path.exists() and not args.rebuild_cache:
                print(f"[cache] Reusing {out_path}")
                continue
            print(f"[cache] Building {out_path} from maps {start}:{stop}")
            x, y = build_wavelet_patches_for_maps(maps[start:stop], nscales=args.nscales,
                                                  nside_cut=args.nside_cut, lmax=3 * args.nside)
            np.savez_compressed(out_path, x=x, y=y)
            print(f"[cache] Saved {out_path}: x={x.shape}, y={y.shape}")
    return shard_paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ShardedLearnletDataset(data.Dataset):
    def __init__(self, shard_paths, split, val_fraction=0.1, seed=42, sigma_max=0.5):
        assert split in ("train", "val")
        self.shard_paths = [Path(p) for p in shard_paths]
        self.sigma_max = float(sigma_max)
        self.entries = []
        rng = np.random.default_rng(seed)
        for shard_idx, p in enumerate(self.shard_paths):
            with np.load(p) as z:
                n = z["x"].shape[0]
            idx = np.arange(n)
            rng.shuffle(idx)
            n_val = max(1, int(round(val_fraction * n)))
            selected = idx[n_val:] if split == "train" else idx[:n_val]
            for local_idx in selected:
                self.entries.append((shard_idx, int(local_idx)))
        self._cache_shard_idx = None
        self._cache_x = None
        self._cache_y = None
        print(f"[dataset:{split}] {len(self.entries)} samples from {len(self.shard_paths)} shards")

    def __len__(self):
        return len(self.entries)

    def _load_shard(self, shard_idx):
        if self._cache_shard_idx == shard_idx:
            return
        z = np.load(self.shard_paths[shard_idx])
        self._cache_x = z["x"]
        self._cache_y = z["y"]
        self._cache_shard_idx = shard_idx

    def __getitem__(self, idx):
        shard_idx, local_idx = self.entries[idx]
        self._load_shard(shard_idx)
        y_scales = torch.from_numpy(self._cache_x[local_idx]).float()    # (nscales, H, W)
        y_map = torch.from_numpy(self._cache_y[local_idx]).float()       # (1, H, W)
        sigma = torch.rand(1).float() * self.sigma_max
        x_noisy = y_scales + torch.randn_like(y_scales) * sigma
        return x_noisy, y_map, sigma.squeeze()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def logcosh_loss(pred, target):
    diff = pred - target
    return (nn.functional.softplus(2.0 * diff) - diff - np.log(2.0)).mean()


def run_one_epoch(model, loader, optimizer, device, train, print_every=0, max_batches=None, epoch=None):
    model.train(train)
    total, n_batches = 0.0, 0
    mode = "train" if train else "val"
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for ibatch, (x_noisy, y_map, sigma) in enumerate(loader, start=1):
            if max_batches is not None and ibatch > max_batches:
                break
            x_noisy = x_noisy.to(device, non_blocking=True)
            y_map = y_map.to(device, non_blocking=True)
            sigma = sigma.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x_noisy, sigma)
            loss = logcosh_loss(pred, y_map)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += float(loss.detach().cpu())
            n_batches += 1
            if print_every and (ibatch == 1 or ibatch % print_every == 0):
                nb_total = len(loader) if max_batches is None else min(len(loader), max_batches)
                ep = "" if epoch is None else f"epoch={epoch:04d} "
                print(f"[{mode}] {ep}batch {ibatch}/{nb_total} loss={float(loss):.6e} "
                      f"running={total / max(n_batches, 1):.6e}", flush=True)
    return total / max(n_batches, 1)


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device: {device}")

    shard_paths = build_or_load_cache(args)
    if not shard_paths:
        raise RuntimeError("No cache shards were created/found.")

    ds_train = ShardedLearnletDataset(shard_paths, "train", args.val_fraction, args.seed, args.sigma_max)
    ds_val = ShardedLearnletDataset(shard_paths, "val", args.val_fraction, args.seed, args.sigma_max)
    loader_train = data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=(device.type == "cuda"))
    loader_val = data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 pin_memory=(device.type == "cuda"))

    model = Learnlet(n_scales=args.nscales, kernel_size=args.kernel_size, filters=args.filters,
                     exact_rec=True, thresh=args.thresh_type, pretrained=False, device=str(device)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_val, patience_counter = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, loader_train, optimizer, device, True,
                                   args.print_every, args.max_train_batches, epoch)
        val_loss = run_one_epoch(model, loader_val, optimizer, device, False,
                                 args.print_every, args.max_val_batches, epoch)
        scheduler.step()
        print(f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.6e} val={val_loss:.6e} "
              f"lr={scheduler.get_last_lr()[0]:.3e}")

        if val_loss < best_val:
            best_val, patience_counter = val_loss, 0
            torch.save({"state_dict": model.state_dict(),
                        "config": {"n_scales": args.nscales, "kernel_size": args.kernel_size,
                                   "filters": args.filters, "exact_rec": True, "thresh": args.thresh_type,
                                   "nside": args.nside, "nside_cut": args.nside_cut,
                                   "sigma_max": args.sigma_max, "components": args.components},
                        "best_val_loss": best_val}, output)
            print(f"  -> saved best model to {output}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    print(f"[train] Finished. Best val loss: {best_val:.6e}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the spherical-wavelet learnlet.")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--components", type=str, nargs="+", default=["gal_ff", "gal_synch", "point_sources"])
    p.add_argument("--dataset_key", type=str, default="auto")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--nscales", type=int, default=5, help="Total scales = 1 coarse + (nscales-1) detail.")
    p.add_argument("--nside_cut", type=int, default=4)
    p.add_argument("--channel_start", type=int, default=0)
    p.add_argument("--channel_end", type=int, default=None)
    p.add_argument("--max_maps_per_file", type=int, default=None)
    p.add_argument("--maps_per_cache_shard", type=int, default=10)
    p.add_argument("--remove_mean", action="store_true", default=True)
    p.add_argument("--no_remove_mean", dest="remove_mean", action="store_false")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no_normalize", dest="normalize", action="store_false")
    p.add_argument("--sigma_max", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--max_train_batches", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--filters", type=int, default=64)
    p.add_argument("--thresh_type", type=str, default="hard", choices=["hard", "soft"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
