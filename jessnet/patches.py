"""HEALPix <-> flat-patch conversion (numba accelerated).

A HEALPix map is split into ``12 * nside_cut**2`` coarse cells; each cell is
unwrapped diagonal-by-diagonal into a square 2-D patch. These patches are the
inputs/outputs of the (planar) spherical-wavelet learnlet.
"""

import os

# This package always ends up with both numba (parallel njit below) and torch
# loaded in the same process. Numba's default parallel threading layer can be
# backed by OpenMP, which then collides with torch's own OpenMP runtime and
# aborts the process with "OMP Error #15: Initializing libomp.dylib, but found
# libomp.dylib already initialized" -- a C-level abort, not a catchable Python
# exception. Force numba onto its 'workqueue' backend (a plain threadpool, no
# OpenMP) instead. Must be set before numba compiles any parallel function, and
# is only honored if not already set by the environment/user.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import healpy as hp
from numba import njit, prange


@njit
def populate_diagonal(to_populate, npix_patch):
    """Fill an (npix_patch, npix_patch) patch from a 1-D array, diagonal by diagonal."""
    n = npix_patch
    patch = np.zeros((n, n), dtype=to_populate.dtype)
    k = 0
    for col in range(n):
        for t in range(col + 1):
            patch[t, col - t] = to_populate[k]
            k += 1
    for row in range(1, n):
        for t in range(n - row):
            patch[row + t, n - 1 - t] = to_populate[k]
            k += 1
    return patch


@njit(parallel=True)
def from_healpix_to_patches_numba(healpix_maps, pixs, ncoarse_pix, npix_patch):
    """Convert HEALPix maps (n_maps, npix) into patches (n_maps, ncoarse, Np, Np)."""
    nb_maps = healpix_maps.shape[0]
    patches = np.zeros((nb_maps, ncoarse_pix, npix_patch, npix_patch), dtype=healpix_maps.dtype)
    for j in prange(nb_maps):
        for i in range(ncoarse_pix):
            flat_patch = healpix_maps[j, pixs[i]]
            patches[j, i] = populate_diagonal(flat_patch, npix_patch)
    return patches


def from_healpix_to_maps_new_numba(healpix_maps, nside_cut=4, verbose=False):
    """Split HEALPix maps into aligned square patches (n_maps, ncoarse, Np, Np)."""
    nside_map = hp.npix2nside(len(healpix_maps[0]))
    ncoarse_pix = hp.nside2npix(nside_cut)
    npix = len(healpix_maps[0]) / ncoarse_pix
    npix_patch = int(np.sqrt(npix))

    if verbose:
        print(f"Cutting HEALPix map with Nside={nside_map} "
              f"onto {ncoarse_pix} patches of {npix_patch}x{npix_patch} pixels...")

    pixs = np.array([np.sort(hp.nest2ring(nside_map,
                                          np.arange(i * npix, (i + 1) * npix, dtype=np.int64)))
                     for i in range(ncoarse_pix)], dtype=np.int64)
    return from_healpix_to_patches_numba(healpix_maps, pixs, ncoarse_pix, npix_patch)


@njit
def patch_to_flat(patch):
    """Unwrap a 2-D patch back into a 1-D array (inverse of populate_diagonal)."""
    n = patch.shape[0]
    total_len = n * (n + 1) // 2 * 2 - n
    result = np.empty(total_len, dtype=patch.dtype)
    k = 0
    for col in range(n):
        for t in range(col + 1):
            result[k] = patch[t, col - t]
            k += 1
    for row in range(1, n):
        for t in range(n - row):
            result[k] = patch[row + t, n - 1 - t]
            k += 1
    return result


@njit(parallel=True)
def from_patches_to_healpix_numba(patches, pixs_out, nb_maps, nb_pix_total):
    """Reassemble patches (n_maps, ncoarse, Np, Np) into HEALPix maps (n_maps, npix)."""
    ncoarse_pix = patches.shape[1]
    healpix_map = np.zeros((nb_maps, nb_pix_total), dtype=patches.dtype)
    for i in prange(ncoarse_pix):
        pix = pixs_out[i]
        for k in range(nb_maps):
            flat_patch = patch_to_flat(patches[k, i])
            healpix_map[k, pix] = flat_patch
    return healpix_map


def from_maps_to_healpix_new_numba(patches):
    """Reassemble square patches (n_maps, ncoarse, Np, Np) into HEALPix maps (n_maps, npix)."""
    ncoarse_pix = patches.shape[1]
    npix_patch = patches.shape[2]
    nb_maps = patches.shape[0]

    npix_total = ncoarse_pix * npix_patch * npix_patch
    nside_out = hp.npix2nside(npix_total)

    pixs_out = np.array([np.sort(hp.nest2ring(nside_out,
                                              np.arange(i * npix_patch * npix_patch,
                                                        (i + 1) * npix_patch * npix_patch, dtype=np.int64)))
                         for i in range(ncoarse_pix)], dtype=np.int64)

    nb_pix_total = hp.nside2npix(nside_out)
    return from_patches_to_healpix_numba(patches, pixs_out, nb_maps, nb_pix_total)
