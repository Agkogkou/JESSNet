"""Multiscale cleaning pipeline: large-scale PCA + per-window JESSNet, with the
windowed recombination. Also the thin `run_jessnet` wrappers that build and run
a JESSNet solver with the paper's default hyperparameters.
"""

import numpy as np
import healpy as hp

from . import harmonics as hpyt
from . import windows as wnd
from .core import JESSNet


# --------------------------------------------------------------------------
# Large-scale PCA
# --------------------------------------------------------------------------
def run_PCA_temp(Xorig, Xmasked=None, NUM=3):
    """PCA foreground removal on the (optionally masked) data.

    The PCA subspace is estimated from `Xmasked` if provided, and the amplitudes
    are fit and removed from `Xorig`.
    """
    print(' Running PCA ...')
    X = np.copy(Xorig) if Xmasked is None else np.copy(Xmasked)

    C = X @ X.conj().T                      # Hermitian covariance
    _, v = np.linalg.eigh(C)               # ascending eigenvalues, orthonormal eigvecs
    Ah = v[:, -NUM:]                        # largest NUM eigenvectors
    S = Ah.conj().T @ Xorig                 # amplitudes fit on the original data
    return Xorig - Ah @ S


def lup_limit_data(cube_lm, lup, lmax, nside):
    """Zero all multipoles above `lup` and return the band-limited map."""
    ell_alm, _ = hp.Alm.getlm(lmax)
    cube_lm_lup = np.zeros_like(cube_lm)
    cube_lm_lup[:, ell_alm <= lup] = cube_lm[:, ell_alm <= lup]
    return hpyt.alm2map_parallel(cube_lm_lup, nside=nside)


# --------------------------------------------------------------------------
# JESSNet solver wrappers (paper defaults)
# --------------------------------------------------------------------------
def _run_solver(perscale, obs_maps, obs_maps_masked, galmask, beam_model, ns,
                K_max, c_wu, nscales=5, alm_in=False, minWuIt=100, wt_input=False):
    Y = obs_maps_masked if obs_maps_masked is not None else obs_maps
    solver = JESSNet(Y, beam_model, ns, galmask,
                     perscale=perscale, minWuIt=minWuIt, c_wu=c_wu, c_ref=c_wu, cwuDec=50,
                     nStd=0.22, useMad=True, nscales=nscales, k=3, K_max=K_max, L1=True, doRw=True,
                     eps=np.array([1e-2, 1e-6, 1e-4]), verb=0, thrEnd=False, nnegA=False, nnegS=False,
                     keepWuRegStr=False, cstWuRegStr=False, alm_in=alm_in)
    solver.run()
    S = solver.S.copy()
    A = solver.A.copy()
    if not wt_input:
        frgrnds = hpyt.convolve(A @ S, beam_model)
        return obs_maps - frgrnds, A, S
    return A, S


def run_jessnet(obs_maps, obs_maps_masked, galmask, beam_model, ns, K_max, c_wu,
                nscales=5, alm_in=False, minWuIt=100, wt_input=False):
    """Run JESSNet on one angular window (perscale=True: masked-aware, oblique A)."""
    return _run_solver(True, obs_maps, obs_maps_masked, galmask, beam_model, ns, K_max, c_wu,
                       nscales, alm_in, minWuIt, wt_input)


def run_monoscale_jessnet(obs_maps, obs_maps_masked, galmask, beam_model, ns, K_max, c_wu,
                          nscales=5, alm_in=False, minWuIt=100, wt_input=False):
    """Run monoscale-JESSNet on the full data (perscale=False, no angular windows,
    no PCA large-scale block). Used as the single-window baseline in the paper."""
    return _run_solver(False, obs_maps, obs_maps_masked, galmask, beam_model, ns, K_max, c_wu,
                       nscales, alm_in, minWuIt, wt_input)


# --------------------------------------------------------------------------
# Hard binning (top-hat windows + sharp assignment)
# --------------------------------------------------------------------------
def run_multiscale_hard(Xlm, Xlm_masked, bl, th, nnside, ell_edges, lup, tau=0.1, num_pca=4,
                        ns=5, K_max=0.9, c_wu=1e-2, nscales=5, n_jobs_conv=5,
                        overlap_sdec=20, galmask=None, Xlm_masked_pca=None, dynamic_overlap=False,
                        minWuIt=100):
    """Multiscale PCA + JESSNet with hard top-hat ell bins.

    If `Xlm_masked_pca` is given, the PCA block trains on it (a less aggressive
    mask). If `dynamic_overlap`, the lower-edge overlap is 5 for the first
    JESSNet block and 10 afterwards (footprint setting).
    """
    lmax = hp.Alm.getlmax(Xlm.shape[1])
    ell, _ = hp.Alm.getlm(lmax)
    pca_block, sdec_blocks = wnd.build_ell_blocks(ell_edges, lmax, lup)
    edge_to_ind = {le: wnd.from_ell_to_nu(bl, le, tau) for le in ell_edges}
    n_freq = Xlm.shape[0]
    Rlm = np.zeros_like(Xlm, dtype=np.complex128)
    if galmask is None:
        galmask = np.ones(hp.nside2npix(nnside), dtype=bool)

    # --- PCA block: ell < l1 ---
    l1 = pca_block[1]
    ind_pca = edge_to_ind[l1]
    mask_pca = (ell >= pca_block[0]) & (ell < pca_block[1])
    y = wnd.apply_beam_ratio(Xlm, bl, ind_pca, n_freq, ell)
    if Xlm_masked is not None:
        pca_src = Xlm_masked_pca if Xlm_masked_pca is not None else Xlm_masked
        y_masked = wnd.apply_beam_ratio(pca_src, bl, ind_pca, n_freq, ell)
        Xmasked_block = y_masked[:, mask_pca]
    else:
        Xmasked_block = None
    Rlm[:, mask_pca] = run_PCA_temp(y[:, mask_pca], Xmasked_block, NUM=num_pca)

    # --- JESSNet blocks ---
    Xlm_actual = Xlm if Xlm_masked is None else Xlm_masked
    for ell_ind, (ell_min, ell_max) in enumerate(sdec_blocks):
        ind_start = edge_to_ind[ell_min]
        ch_idx = np.arange(ind_start, n_freq)
        ov = (5 if ell_ind == 0 else 10) if dynamic_overlap else overlap_sdec
        mask_fit = wnd.get_mask_from_block(ell, ell_min, ell_max, overlap_low=ov)

        temp_alm_orig = np.zeros((len(ch_idx), Xlm_actual.shape[1]), dtype=np.complex128)
        temp_alm_orig[:, mask_fit] = Xlm[np.ix_(ch_idx, mask_fit)]
        temp_alm_block = np.zeros((len(ch_idx), Xlm_actual.shape[1]), dtype=np.complex128)
        temp_alm_block[:, mask_fit] = Xlm_actual[np.ix_(ch_idx, mask_fit)]

        print(f"Running JESSNet on ell=[{ell_min},{ell_max}) with {len(ch_idx)} channels")
        A_sdec, S_sdec = run_jessnet(temp_alm_block, None, galmask, bl[ch_idx], ns=ns, K_max=K_max,
                                     c_wu=c_wu, nscales=nscales, alm_in=True, wt_input=True,
                                     minWuIt=minWuIt)

        frg_rec = hpyt.convolve_parallel(A_sdec @ S_sdec, th[ch_idx], ell_max, n_jobs=n_jobs_conv)
        frg_rec_lm = hpyt.map2alm(frg_rec, lmax=lmax)
        recHI_block = temp_alm_orig - frg_rec_lm

        mask_out = (ell >= ell_min) & (ell < ell_max)
        Rlm[np.ix_(ch_idx, mask_out)] = recHI_block[:, mask_out]

    return Rlm


# --------------------------------------------------------------------------
# Windowed binning (cosine / wavelet windows + sum recombination)
# --------------------------------------------------------------------------
def run_multiscale_windowed(Xlm, Xlm_masked, bl, th, nnside, windows, pca_window_ids, tau=0.1,
                            channel_selection='weighted', num_pca=4, ns=5, K_max=0.9, c_wu=1e-2,
                            nscales_sdec=5, n_jobs_conv=5, galmask=None,
                            filter_threshold=3e-2, renormalize_available_windows=False,
                            Xlm_masked_pca=None, minWuIt=100):
    """Multiscale PCA + JESSNet with smooth (cosine/wavelet) windows.

    Each scale is X_j = W_j(ell) X; the cleaned scale is added to Rlm (analysis
    by W_j, synthesis by summation). `pca_window_ids` are the window indices
    cleaned with PCA (the coarse/large-scale window). `Xlm_masked_pca`, if given,
    is the (less aggressively) masked dataset used to train the PCA window(s).
    """
    lmax = hp.Alm.getlmax(Xlm.shape[1])
    ell, _ = hp.Alm.getlm(lmax)
    if windows.shape[1] != lmax + 1:
        raise ValueError(f'windows lmax={windows.shape[1]-1}, but Xlm lmax={lmax}')
    if galmask is None:
        galmask = np.ones(hp.nside2npix(nnside), dtype=bool)

    Xlm_actual = Xlm if Xlm_masked is None else Xlm_masked
    Xlm_pca_train = Xlm_masked_pca if Xlm_masked_pca is not None else Xlm_actual

    Rlm = np.zeros_like(Xlm, dtype=np.complex128)
    Norm = np.zeros_like(Xlm.real, dtype=float)

    for j, W_l in enumerate(windows):
        W_l = np.asarray(W_l, dtype=float)
        W_lm = W_l[ell]
        maxW = np.max(np.abs(W_lm))
        if maxW <= 0:
            print(f'Skipping window {j}: zero window.')
            continue
        active = np.abs(W_lm) > filter_threshold * maxW
        if not np.any(active):
            print(f'Skipping window {j}: empty active support.')
            continue

        ell_active = ell[active]
        ell_max_eff = int(ell_active.max())
        ch_idx = wnd.channels_from_window(bl=bl, W_l=W_l, tau=tau, mode=channel_selection)
        print(f"Window {j}: ell in [{int(ell_active.min())},{ell_max_eff}], "
              f"channels={len(ch_idx)}, method={'PCA' if j in pca_window_ids else 'JESSNet'}")

        Xj_orig = np.zeros((len(ch_idx), Xlm.shape[1]), dtype=np.complex128)
        Xj_orig[:, active] = Xlm[np.ix_(ch_idx, active)] * W_lm[active][None, :]

        if j in pca_window_ids:
            if Xlm_masked is not None:
                Xtrain_active = Xlm_pca_train[np.ix_(ch_idx, active)] * W_lm[active][None, :]
            else:
                Xtrain_active = None
            rec_active = run_PCA_temp(Xj_orig[:, active], Xtrain_active, NUM=num_pca)
            rec_j = np.zeros_like(Xj_orig)
            rec_j[:, active] = rec_active
        else:
            Xj_block = np.zeros((len(ch_idx), Xlm.shape[1]), dtype=np.complex128)
            Xj_block[:, active] = Xlm_actual[np.ix_(ch_idx, active)] * W_lm[active][None, :]
            A_sdec, S_sdec = run_jessnet(Xj_block, None, galmask, bl[ch_idx], ns=ns, K_max=K_max,
                                         c_wu=c_wu, nscales=nscales_sdec,
                                         alm_in=True, wt_input=True, minWuIt=minWuIt)
            frg_rec = hpyt.convolve_parallel(A_sdec @ S_sdec, th[ch_idx], ell_max_eff, n_jobs=n_jobs_conv)
            frg_rec_lm = hpyt.map2alm(frg_rec, lmax=lmax)
            rec_j = np.zeros_like(Xj_orig)
            rec_j[:, active] = Xj_orig[:, active] - frg_rec_lm[:, active]

        Rlm[ch_idx, :] += rec_j
        Norm[ch_idx, :] += np.abs(W_lm)[None, :]

    if renormalize_available_windows:
        good = Norm > 1e-12
        Rlm[good] /= Norm[good]

    return Rlm
