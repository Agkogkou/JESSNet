# Learnlet weights

The trained spherical-wavelet learnlet weights used in the paper are distributed
via **Zenodo**:

> **[ Zenodo DOI / link — TO BE ADDED ]**

Place the file here:

```
weights/
└── learnlet_sphere_64_5_sc5_fg.pth
```

and point `core.WEIGHT_PATH` (in `scripts/run_jessnet.py`) at it. The file name
encodes the architecture: `64` filters, kernel size `5`, `sc5` = 5 spherical
scales (1 coarse + 4 detail). These must match `core.LEARNLET_NSCALES`,
`LEARNLET_FILTERS`, and `LEARNLET_KERNEL_SIZE`.

To train your own weights, see `training/train_learnlet.py`. Weight files are
**not** tracked by git (see `.gitignore`).
