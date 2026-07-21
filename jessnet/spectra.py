"""Power-spectrum diagnostics used to evaluate the cleaning: angular power
spectra (C_ell) and the radial/frequency power spectrum P(k_nu)."""

import numpy as np
import healpy as hp
import scipy.fftpack
from joblib import Parallel, delayed


# --------------------------------------------------------------------------
# Angular power spectrum
# --------------------------------------------------------------------------
def Cell(map):
    """Angular power spectrum of a single HEALPix map."""
    cl = hp.anafast(map)
    ell = np.arange(len(cl))
    return ell, cl


def compute_Cell_one_band(nui, arr):
    _, cl = Cell(arr[nui, :])
    return cl


# --------------------------------------------------------------------------
# Radial (frequency) power spectrum P(k_nu)
# --------------------------------------------------------------------------
def clustering_nu(field_array, indexes_los, nu_ch, verbose=False):
    """FFT along the frequency axis over a set of lines of sight."""
    T_field = field_array[:, indexes_los]
    del field_array

    nlos = len(indexes_los)
    if verbose:
        print(f'using {nlos} LoS')
    del indexes_los

    dims = len(nu_ch)
    dnu = abs(nu_ch[-1] - nu_ch[-2])
    if verbose:
        print(f'each divided into {dims} cells of {dnu} MHz')

    deltaT = np.array([T_field[:, ipix] for ipix in range(nlos)])
    del T_field

    if verbose:
        print('\nFFT the overdensity temperature field along LoS')
    delta_k = scipy.fftpack.fftn(deltaT, overwrite_x=True, axes=1)
    delta_k *= dnu
    del deltaT

    delta_k_auto = np.absolute(delta_k) ** 2
    if verbose:
        print('done!\n')
    return dims, dnu, delta_k_auto


def doing_Pk1D(dims, dnu, delta_k_auto):
    """Bin the FFT modulus into a 1-D radial power spectrum."""
    modes = np.arange(dims, dtype=np.float64)
    middle = int(dims / 2)
    indexes = np.where(modes > middle)[0]
    modes[indexes] = modes[indexes] - dims
    k = modes * (2.0 * np.pi / (dnu * dims))
    k = np.absolute(k)
    del indexes, modes

    k_bins = np.linspace(0, middle, middle + 1) * (2.0 * np.pi / (dnu * dims))

    k_modes = np.histogram(k, bins=k_bins)[0]
    k_bin = np.histogram(k, bins=k_bins, weights=k)[0] / k_modes

    delta_k2_stacked = np.mean(delta_k_auto, dtype=np.float64, axis=0)

    Pk_mean = np.histogram(k, bins=k_bins, weights=delta_k2_stacked)[0]
    Pk_mean = Pk_mean / (dnu * dims * k_modes)
    del delta_k2_stacked

    Pk_1D = np.transpose([k_bin[1:], Pk_mean[1:]])
    return Pk_1D


def plot_nuPk(fmap, indexes_los, nu_ch, verbose=False):
    """Frequency power spectrum P(k_nu); returns (k_nu, P)."""
    Pk_1D = doing_Pk1D(*clustering_nu(fmap, indexes_los, nu_ch))
    if verbose:
        print("k_nu [MHz^-1] vs P [mK^2 MHz]")
    return Pk_1D[:, 0], Pk_1D[:, 1]


# --------------------------------------------------------------------------
# Footprint / Galactic-mask power spectra
# --------------------------------------------------------------------------
def post_GalMasking(masks, mask_names, freqs, input_map, n_jobs=20):
    """Compute masked angular + radial power spectra over a set of named masks."""
    nfreq, npix = input_map.shape
    P_maskeds, cl_maskeds = {}, {}
    ell_masked = None
    k_masked = None
    idx = np.arange(npix)

    for name, m in zip(mask_names, masks):
        mask = m.astype(bool)
        k_masked, P_masked = plot_nuPk(input_map, idx[mask], freqs)
        P_maskeds[name] = P_masked

        input_map_mask = np.array(input_map, copy=True)
        input_map_mask[:, ~mask] = hp.UNSEEN

        cl_rows = Parallel(n_jobs=n_jobs, prefer="processes", mmap_mode="r", max_nbytes="10M")(
            delayed(compute_Cell_one_band)(i, input_map_mask) for i in range(nfreq))

        cl_maskeds[name] = np.stack(cl_rows, axis=0)
        ell_masked, _ = Cell(input_map_mask[0, :])

    return k_masked, P_maskeds, ell_masked, cl_maskeds
