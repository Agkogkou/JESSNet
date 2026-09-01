# Data

The input simulations used in the paper (and the survey-footprint / Planck mask
files) are distributed via **Zenodo**:

> [doi.org/10.5281/zenodo.20341922](https://doi.org/10.5281/zenodo.20341922)

After downloading, place the files directly here (Zenodo does not preserve a
folder structure) so the default paths in `scripts/run_jessnet.py` resolve:

```
data/
├── sim_CoLoRe_1.0MHz_nside256_gaussian_oscillating.hd5     # input observation + HI truth cubes
├── footprint_intermediate_plus_Planck70_nside256_apodized.npy
└── footprint_intermediate_plus_Planck70_nside256_apodized_shrunk.npy
```

## Expected contents of the input `.hd5`

| dataset            | shape            | description                                   |
|--------------------|------------------|-----------------------------------------------|
| `frequencies`      | `(n_freq,)`      | frequency channels [MHz]                      |
| `Obs_conv_noise`   | `(n_freq, npix)` | beam-convolved observation (fg + HI + noise)  |
| `HI_conv_noise`    | `(n_freq, npix)` | beam-convolved HI ground truth                |

Maps are HEALPix (`RING` ordering), `nside = 256`.

These files are large and are intentionally **not** tracked by git (see
`.gitignore`).
