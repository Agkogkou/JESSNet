"""Angular-window construction and per-window channel selection.

Three window families are supported: hard top-hat bands, cosine-tapered
overlapping bands, and starlet-like wavelet bands. For the smooth families the
windows form a partition of unity so the cleaned windowed contributions can be
recombined by summation.
"""

import numpy as np

from . import harmonics as hpyt


# --------------------------------------------------------------------------
# Multipole blocks (hard binning)
# --------------------------------------------------------------------------
def from_ell_to_nu(bl, ell_scale, tau):
    """First channel index whose beam response at `ell_scale` is >= tau."""
    idx = np.where(bl[:, ell_scale] >= tau)[0]
    if len(idx) == 0:
        raise ValueError(f"No channel satisfies bl[:, {ell_scale}] >= {tau}")
    return idx[0]


def build_ell_blocks(ell_edges, lmax, lup):
    """PCA block (0, l1) and JESSNet blocks [(l1,l2), ..., (l_last, lup+1)]."""
    ell_edges = sorted(set(int(x) for x in ell_edges))
    if len(ell_edges) == 0:
        raise ValueError("ell_edges must contain at least one multipole boundary.")
    if ell_edges[0] <= 0:
        raise ValueError("All ell_edges must be > 0.")
    if ell_edges[-1] > lmax:
        raise ValueError(f"Last ell edge {ell_edges[-1]} exceeds lmax={lmax}.")

    pca_block = (0, ell_edges[0])
    sdec_blocks = [(ell_edges[i], ell_edges[i + 1]) for i in range(len(ell_edges) - 1)]
    sdec_blocks.append((ell_edges[-1], lup + 1))
    return pca_block, sdec_blocks


def get_mask_from_block(ell_array, ell_min, ell_max, overlap_low=0):
    """Top-hat mask ell_min - overlap_low <= ell < ell_max."""
    low = max(0, ell_min - overlap_low)
    return (ell_array >= low) & (ell_array < ell_max)


def apply_beam_ratio(Xlm, bl, ind_pca, n_freq, ell):
    """Reconvolve channels >= ind_pca to the common beam bl[ind_pca] (vectorized)."""
    y = np.copy(Xlm)
    ratios = bl[ind_pca] / bl[ind_pca:n_freq]      # (n_freq-ind_pca, lmax+1)
    y[ind_pca:n_freq] = Xlm[ind_pca:n_freq] * ratios[:, ell]
    return y


# --------------------------------------------------------------------------
# Window construction
# --------------------------------------------------------------------------
def make_hard_ell_windows(lmax, ell_edges, lup):
    """Hard top-hat windows (partition of unity below lup)."""
    ell_grid = np.arange(lmax + 1)
    edges = [0] + list(ell_edges) + [lup + 1]
    windows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        W = np.zeros(lmax + 1, dtype=float)
        W[(ell_grid >= lo) & (ell_grid < hi)] = 1.0
        windows.append(W)
    return np.asarray(windows)


def smoothstep_cosine(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * x))


def normalize_windows_sum(windows):
    """Normalize windows so that sum_j W_j(ell) = 1 wherever there is support."""
    windows = np.asarray(windows, dtype=float)
    Wsum = np.sum(windows, axis=0)
    good = Wsum > 0
    out = np.zeros_like(windows)
    out[:, good] = windows[:, good] / Wsum[good][None, :]
    return out


def make_cosine_ell_windows(lmax, ell_edges, lup, width=5):
    """Cosine-tapered overlapping windows, normalized to a partition of unity."""
    ell_grid = np.arange(lmax + 1)
    edges = list(ell_edges)
    nwin = len(edges) + 1
    windows = []
    for i in range(nwin):
        W = np.ones(lmax + 1, dtype=float)
        if i > 0:                                       # left transition 0 -> 1
            left = edges[i - 1]
            W *= smoothstep_cosine((ell_grid - (left - width)) / (2.0 * width))
        if i < len(edges):                              # right transition 1 -> 0
            right = edges[i]
            W *= (1.0 - smoothstep_cosine((ell_grid - (right - width)) / (2.0 * width)))
        if i == nwin - 1:                               # upper cutoff for last window
            W *= (1.0 - smoothstep_cosine((ell_grid - (lup - width)) / (2.0 * width)))
        else:
            W[ell_grid > lup] = 0.0
        windows.append(W)
    return normalize_windows_sum(np.asarray(windows))


def make_wavelet_ell_windows(lmax, n_wavelet_scales, lup=None, renormalize=True):
    """Starlet-like wavelet windows (nscales+1 bands); last band is the coarse scale."""
    wt_filters = hpyt.get_wt_filters(lmax, n_wavelet_scales)
    windows = wt_filters.T.copy()
    if lup is not None:
        ell_grid = np.arange(lmax + 1)
        windows[:, ell_grid > lup] = 0.0
    if renormalize:
        windows = normalize_windows_sum(windows)
    return windows


def save_window_diagnostic_plot(windows, output_file):
    """Optional diagnostic plot of the angular windows W_j(ell)."""
    try:
        import matplotlib.pyplot as plt
        lmax = windows.shape[1] - 1
        ell_grid = np.arange(lmax + 1)
        plt.figure(figsize=(5, 3), dpi=150)
        for j in range(windows.shape[0]):
            plt.plot(ell_grid, windows[j], label=f'window {j}')
        plt.xscale('log')
        plt.xlabel(r'Multipole $\ell$')
        plt.ylabel(r'$W_j(\ell)$')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    except Exception as e:
        print(f'Could not save window diagnostic plot: {repr(e)}')


# --------------------------------------------------------------------------
# Channel selection per window
# --------------------------------------------------------------------------
def wavelet_weighted_beam_response(bl, W_l):
    """Window-weighted beam response per channel: sum_l W^2 B / sum_l W^2."""
    w = W_l ** 2
    wsum = np.sum(w)
    if wsum <= 0:
        raise ValueError('Window has zero norm.')
    return np.sum(bl * w[None, :], axis=1) / wsum


def effective_ell_for_window(W_l, frac=0.95):
    """Multipole containing `frac` of the cumulative W_l^2 support."""
    ell_grid = np.arange(len(W_l))
    w = W_l ** 2
    if np.sum(w) <= 0:
        raise ValueError('Window has zero norm.')
    cdf = np.cumsum(w) / np.sum(w)
    return int(ell_grid[np.searchsorted(cdf, frac)])


def channels_from_window(bl, W_l, tau=0.1, mode='weighted'):
    """Select channels for one window.
    'weighted': window-weighted beam response >= tau;
    'effective_ell': B_nu(ell_eff) >= tau; 'all': keep every channel."""
    n_freq = bl.shape[0]
    if mode == 'all':
        return np.arange(n_freq)
    if mode == 'weighted':
        Bbar = wavelet_weighted_beam_response(bl, W_l)
        ch_idx = np.where(Bbar >= tau)[0]
    elif mode == 'effective_ell':
        ell_eff = np.clip(effective_ell_for_window(W_l, frac=0.95), 0, bl.shape[1] - 1)
        ch_idx = np.where(bl[:, ell_eff] >= tau)[0]
    else:
        raise ValueError(f'Unknown channel-selection mode: {mode}')
    if len(ch_idx) == 0:
        raise ValueError('No channel satisfies the beam threshold for this window.')
    return ch_idx
