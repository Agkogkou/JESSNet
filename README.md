# JESSNet — Joint dEconvolution and Sparse Separation Network

Beam-aware, multiscale foreground cleaning for 21 cm intensity mapping.

JESSNet builds on DecGMCA / SDecGMCA (joint deconvolution and blind source
separation in spherical-harmonic space) and extends it in three ways:

1. **Multiscale angular windows** — the data are split into angular windows; an
   effective mixing matrix is estimated independently in each window, using only
   the frequency channels that carry reliable angular information at that scale.
2. **Learned sparse regularization** — the fixed starlet thresholding of
   SDecGMCA is replaced by a **spherical-wavelet learnlet**, a learned
   proximal-like denoiser applied to multiscale patch cubes on the sphere.
3. **Mask-constrained reconstruction** — the source estimates are restricted to
   the observed sky region, so the method applies to Galactic masks and survey
   footprints.

---

## Installation

```bash
git clone https://github.com/Agkogkou/JESSNet
cd JESSNet
pip install -e .          # or: pip install -r requirements.txt
```

Requires Python ≥ 3.9. The learnlet step (`torch`) runs on CUDA or Apple Silicon
(MPS) GPUs when available, and falls back to CPU otherwise.

## Data and weights (Zenodo)

The input simulation, the mask/footprint files, and the trained learnlet weights
are distributed via Zenodo: [doi.org/10.5281/zenodo.20341922](https://doi.org/10.5281/zenodo.20341922)
(see `data/README.md` and `weights/README.md` for the expected layout):

```
data/    sim_*.hd5, footprint_*.npy            # from Zenodo (doi.org/10.5281/zenodo.20341922)
weights/ learnlet_sphere_64_5_sc5_fg.pth       # from Zenodo (doi.org/10.5281/zenodo.20341922)
```

## Quick start

Edit the `CONFIG` block at the top of `scripts/run_jessnet.py` (paths, sky mode,
window mode, learnlet knob), then:

```bash
python scripts/run_jessnet.py
```

This runs the monoscale-JESSNet baseline (no angular windows, no PCA block)
and the full multiscale JESSNet cleaning (PCA on the largest angular scales +
per-window JESSNet), and saves the angular (`C_ell`) and radial (`P(k_nu)`)
power spectra for `HI`, `JESSNet` (multiscale), and `monoscaleJESSNet`
(baseline) into `outputs/`.

### Key options

| option | meaning |
|---|---|
| `mask_galactic_plane` | `0` full sky, `1` survey footprint |
| `window_mode` | `'hard'` (top-hat), `'cosine'` (tapered), `'wavelet'` (starlet) angular windows |
| `ell_edges`, `lup` | window boundaries and maximum reconstructed multipole |
| `core.LEARNLET_K` | learnlet noise level `sigma = k · MAD(source)` — **the main threshold knob** |
| `core.WEIGHT_PATH`, `core.LEARNLET_NSCALES` | learnlet weights and number of spherical scales (must match the file) |
| `core.PROFILE` | print a per-section wall-clock breakdown of the solver |

The same trained weights work for **both** full-sky and footprint runs: each
source is normalized to unit variance over the observed region before the
learnlet, matching the unit-variance normalization used at training time.

## Parameters to reproduce the results of Gkogkou et al. (2026)

The `CONFIG` block in `scripts/run_jessnet.py` already ships with these
values; this section is a quick reference. Full-sky and footprint runs share
almost every setting — only two differ.

**Differs between full-sky and footprint:**

| parameter | full-sky | footprint |
|---|---|---|
| `mask_galactic_plane` | `0` | `1` |
| `sigma_mask` | `5` | `3` |

**Shared:**

| parameter | value |
|---|---|
| `nnside` | `256` |
| `lup` | `256` (= `nnside`) |
| `bright_mask` | `True` |
| `strategy_mask` | `1` (wavelet masking) |
| `degraded` | `False` |
| `oscillating` | `True` |
| `beam` | `0` (Gaussian) |
| `window_mode` | `'cosine'` |
| `ell_edges` | `[20, 100]` |
| `taper_width` | `10` |
| `channel_selection` | `'weighted'` |
| `tau` | `0.1` |
| `ns_PCA` | `4` |
| `ns_JESSNet` | `5` |
| `K_max` | `0.6` |
| `c_wu` | `1e-2` |
| `renormalize_available_windows` | `False` |
| `core.LEARNLET_NSCALES` | `5` |
| `core.LEARNLET_NSIDE_CUT` | `4` |
| `core.LEARNLET_K` | `0.5` |
| `core.LEARNLET_BATCH_SIZE` | `400` |

One value lives outside the `CONFIG` block, as a function default in
`jessnet/pipeline.py`'s `run_multiscale_windowed`: **`filter_threshold =
3e-2`**. It bounds how far into its own cosine-taper tail a window still
takes responsibility for cleaning before handing that multipole range off to
the neighboring window, which by construction has most of the weight there.
The shipped default already matches this value, so no action is needed unless
you call `run_multiscale_windowed` directly with an explicit override.

## Library use

```python
import jessnet.core as core
from jessnet import beams, harmonics as hpyt
from jessnet.pipeline import run_multiscale_windowed
from jessnet.windows import make_cosine_ell_windows

core.WEIGHT_PATH = "weights/learnlet_sphere_64_5_sc5_fg.pth"
core.LEARNLET_K  = 0.5

# Xlm: (n_freq, n_alm) observed harmonic coefficients; bl, th: beam model; ...
win = make_cosine_ell_windows(lmax, ell_edges=[20, 100], lup=256, width=10)
Rlm = run_multiscale_windowed(Xlm, Xlm_masked, bl, th, nside, win, pca_window_ids=[0])
```

## Repository layout

```
jessnet/
  core.py          JESSNet solver class (beam-aware source/mixing updates + learnlet)
  learnlet.py      spherical-wavelet learnlet network
  pipeline.py      multiscale PCA + JESSNet drivers and run wrappers
  windows.py       angular windows (hard/cosine/wavelet) + channel selection
  harmonics.py     spherical-harmonic transforms + isotropic spherical wavelets
  patches.py       HEALPix <-> flat-patch conversion (numba)
  beams.py         Gaussian beam model
  preprocessing.py run settings + bright-source masking
  spectra.py       angular (C_ell) and radial (P(k_nu)) power-spectrum diagnostics
  utils.py         robust MAD estimate + (optional) ground-truth evaluation
scripts/run_jessnet.py    end-to-end analysis (edit CONFIG, then run)
training/train_learnlet.py  train the spherical-wavelet learnlet
```

## Training the learnlet

```bash
python training/train_learnlet.py \
  --input_dir /path/to/fg_training --components gal_ff gal_synch point_sources \
  --output weights/learnlet_sphere_64_5_sc5_fg.pth \
  --cache_dir /path/to/cache --nside 256 --nscales 5 --nside_cut 4 \
  --sigma_max 0.5 --epochs 1000 --device cuda:0
```

Inference feeds `sigma = k · MAD(source)`; `MAD` of a unit-variance field ≈ 1, so
`k` and the training `sigma_max` are on the same scale. Train with
`--sigma_max ≳ max(k)` you intend to use (e.g. `--sigma_max 0.5` for `k ≲ 0.5`).

## Citation

If you use JESSNet, please cite:

```bibtex
@article{jessnet,
  author  = {TO BE ADDED},
  title   = {JESSNet: Joint dEconvolution and Sparse Separation Network},
  year    = {2026},
  journal = {TO BE ADDED}
}
```

JESSNet builds on SDecGMCA (Carloni Gertosio & Bobin 2021), DecGMCA
(Jiang et al. 2017), L-GMCA (Bobin et al. 2013), and learnlets (Ramzi et al.
2021; Bonjean et al. 2026).
