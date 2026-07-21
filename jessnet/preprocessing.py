"""Input preprocessing: run-name settings and masking of extremely bright HI
(point-like) sources prior to the foreground separation."""

import numpy as np
from scipy.stats import sigmaclip
from scipy.ndimage import convolve1d
from joblib import Parallel, delayed


def make_settings(pol, degraded, beam, interp=True, bright_mask=False, oscillating_beam=False,
                  strategy_mask=0, mask_galactic_plane=0):
    """Build the descriptive key strings used to name output folders/files."""
    pol = '' if pol else '_no_pol'
    degraded, _ = ('_degraded', 3) if degraded else ('', 4)
    beam = '_gaussian' if beam == 0 else '_modairy'
    interp = '_interp' if interp else ''
    bright_mask = '_masked' if bright_mask else ''
    oscillating_beam = '_oscillating' if oscillating_beam else ''

    strategy_mask_key = '_polynomialMasking' if strategy_mask == 0 else '_waveletMasking'
    mask_galactic_plane_key = '' if mask_galactic_plane == 0 else '_withGalMask'

    return (pol, degraded, beam, interp, bright_mask, oscillating_beam,
            strategy_mask_key, mask_galactic_plane_key)


# --------------------------------------------------------------------------
# 1-D starlet transform (along frequency), used by the wavelet-based masking
# --------------------------------------------------------------------------
def starlet_transform(signal, J=4):
    h = np.array([1, 4, 6, 4, 1]) / 16.0  # B3-spline
    approx = signal.copy()
    coeffs = []
    for j in range(J):
        step = 2 ** j
        h_j = np.zeros((len(h) - 1) * step + 1)
        h_j[::step] = h
        smoothed = convolve1d(approx, h_j, mode='reflect')
        detail = approx - smoothed
        coeffs.append(detail)
        approx = smoothed
    coeffs.append(approx)
    return coeffs


def starlet_inverse(coeffs):
    return np.sum(coeffs, axis=0)


def _process_healpix_column(col_data, sigma_mask):
    """Mask outliers on the finest 1-D starlet scales of a single line of sight."""
    coeffs = np.array(starlet_transform(col_data))

    small_sc_noborder = coeffs[0][2:-2]
    sigma_nmad = 1.4826 * np.median(np.abs(small_sc_noborder - np.median(small_sc_noborder)))
    small_sc_noborder[small_sc_noborder > sigma_mask * sigma_nmad] = 0
    coeffs[0][2:-2] = small_sc_noborder

    small_sc_noborder = coeffs[1][4:-4]
    sigma_nmad = 1.4826 * np.median(np.abs(small_sc_noborder - np.median(small_sc_noborder)))
    small_sc_noborder[small_sc_noborder > sigma_mask * sigma_nmad] = 0
    coeffs[1][4:-4] = small_sc_noborder

    return starlet_inverse(coeffs)


def masking_input_data_healpix_wavelet(mock_cube, sigma_mask=5, n_jobs=-1, verbose=0):
    """Wavelet-based bright-source masking (per line of sight), parallelized."""
    print('Masking the extremely bright HI sources from the input HEALPix maps with wavelets...')
    n_cols = mock_cube.shape[1]
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_process_healpix_column)(mock_cube[:, i], sigma_mask) for i in range(n_cols))
    return np.column_stack(results)


def masking_input_data_healpix_polyn(mock_cube, freqs, n=6, sigma_mask=5):
    """Polynomial-fit bright-source masking (per line of sight)."""
    print('Masking the extremely bright HI sources from the input HEALPix maps (polynomial)...')

    rec_polyn = np.copy(mock_cube)
    smooth_fit = np.zeros_like(rec_polyn)
    filtered_cube = np.copy(mock_cube)

    for i in range(mock_cube.shape[1]):
        z = np.polyfit(freqs, mock_cube[:, i], n)
        rec_polyn[:, i] = mock_cube[:, i] - np.polyval(z, freqs)

    stds, means = np.zeros(rec_polyn.shape[1]), np.zeros(rec_polyn.shape[1])
    for i in range(rec_polyn.shape[1]):
        j = sigmaclip(rec_polyn[:, i], 5).clipped
        stds[i] = np.std(j)
        means[i] = np.mean(j)

    mask = abs(rec_polyn - means[None, :]) < sigma_mask * stds[None, :]

    for i in range(mock_cube.shape[1]):
        z = np.polyfit(freqs[mask[:, i]], mock_cube[:, i][mask[:, i]], n)
        smooth_fit[:, i] = np.polyval(z, freqs)

    for ifr in range(freqs.shape[0]):
        indhp = np.where(~mask[ifr])
        filtered_cube[ifr, indhp] = smooth_fit[ifr, indhp]

    return filtered_cube, mask
