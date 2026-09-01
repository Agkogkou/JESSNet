"""End-to-end JESSNet foreground-cleaning run.

Reproduces the full-sky and survey-footprint analyses of the paper: multiscale
PCA + JESSNet cleaning, a full-data (no-PCA) baseline, and the angular/radial
power-spectrum diagnostics.

Edit the CONFIG block below (paths, window mode, learnlet knob) and run:

    python scripts/run_jessnet.py

The input simulation and the trained learnlet weights are distributed via
Zenodo (see README / data/ and weights/).

NOTE: the harmonic transforms use a multiprocessing.ProcessPoolExecutor (see
jessnet/harmonics.py). On macOS/Windows (the 'spawn' start method), every
worker process re-imports this file, so the analysis body must live inside
`if __name__ == "__main__":` -- otherwise each spawned worker re-triggers the
same calls itself, recursively. Keep this guard if you edit the script.
"""

import os
import sys
import time

import numpy as np
import healpy as hp
import h5py
from joblib import Parallel, delayed

# make the `jessnet` package importable when running from the repo without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jessnet.core as core
from jessnet import beams, harmonics as hpyt, preprocessing, spectra, windows as wnd
from jessnet.pipeline import (run_monoscale_jessnet, run_multiscale_hard,
                              run_multiscale_windowed, lup_limit_data)

# =====================================================================
#                              CONFIG
# =====================================================================
# --- paths (edit these) ---
INPUT_FILE = 'data/sim_CoLoRe_1.0MHz_nside256_gaussian_oscillating.hd5'
FOOTPRINTS_DIR = 'data/'
OUTPUT_DIR = 'outputs/'
CASE_NAME = 'jessnet_run'

# --- sky / analysis setup ---
nnside = 256
mask_galactic_plane = 0        # 0 = full sky, 1 = survey footprint
bright_mask = True             # mask extremely bright HI sources
strategy_mask = 1              # 0 = polynomial masking, 1 = wavelet masking
sigma_mask = 5                 # bright-source masking threshold (sigma)
degraded = False
oscillating = True             # oscillating Gaussian beam
beam = 0                       # 0 = Gaussian beam

# --- multiscale / windows ---
window_mode = 'cosine'         # 'hard' | 'cosine' | 'wavelet'
ell_edges = [20, 100]
lup = nnside                   # reconstruct multipoles up to lup
taper_width = 10               # cosine mode
n_wavelet_scales = 2           # wavelet mode
channel_selection = 'weighted' # 'weighted' | 'effective_ell' | 'all' (cosine/wavelet)
renormalize_available_windows = False
overlap_sdec = 5               # hard mode lower-edge overlap
tau = 0.1                      # channel-selection beam threshold
ns_PCA = 4                     # PCA modes for the large-scale window
ns_JESSNet = 5                 # number of effective components per JESSNet window
K_max = 0.6
c_wu = 1e-2

# --- spherical-wavelet learnlet ---
core.WEIGHT_PATH = 'weights/learnlet_sphere_64_5_sc5_fg.pth'
core.LEARNLET_NSCALES = 5      # must match the trained weights (..._sc5)
core.LEARNLET_NSIDE_CUT = 4
core.LEARNLET_K = 0.5          # sigma = k * MAD(source); the tunable threshold knob
core.LEARNLET_BATCH_SIZE = 400
core.PROFILE = False           # True -> print a per-section timing breakdown

# footprint-specific overrides (applied automatically when mask_galactic_plane==1)
if mask_galactic_plane == 1:
    sigma_mask = 3
    ns_PCA = 3
    lup = 250

# =====================================================================
def main():
    start = time.time()
    sky_type = 'fullsky' if mask_galactic_plane == 0 else 'footprint'
    results_output_path = os.path.join(OUTPUT_DIR, sky_type, f'{CASE_NAME}_{window_mode}') + '/'
    os.makedirs(results_output_path, exist_ok=True)
    print(f'window_mode = {window_mode} | sky = {sky_type} | ell_edges = {ell_edges} up to {lup}')
    print(f'Results will be saved in: {results_output_path}')

    # --- load data ---
    file = h5py.File(INPUT_FILE, 'r')
    freqs = np.array(file['frequencies'])
    print(f'{len(freqs)} channels, {min(freqs)}-{max(freqs)} MHz, d_nu={freqs[1]-freqs[0]} MHz')

    obs_maps = np.array(file['Obs_conv_noise'])
    obs_maps -= np.mean(obs_maps, axis=1, keepdims=True)
    hi_signal_smoothed = np.array(file['HI_conv_noise'])
    hi_signal_smoothed -= np.mean(hi_signal_smoothed, axis=1, keepdims=True)

    # --- beam model ---
    th, ell_beam_model, bl = beams.gen_beam_model(degraded, oscillating, freqs, nnside, A_beam=0.5, T_beam=20)

    # --- bright-source masking ---
    if bright_mask:
        if strategy_mask == 0:
            obs_maps_masked, _ = preprocessing.masking_input_data_healpix_polyn(obs_maps, freqs, n=6, sigma_mask=sigma_mask)
            obs_maps_masked_pca = obs_maps_masked
        else:
            obs_maps_masked = preprocessing.masking_input_data_healpix_wavelet(obs_maps, sigma_mask=sigma_mask)
            if mask_galactic_plane == 1:
                obs_maps_masked_pca = preprocessing.masking_input_data_healpix_wavelet(obs_maps, sigma_mask=4)
            else:
                obs_maps_masked_pca = obs_maps_masked
    else:
        obs_maps_masked = None
        obs_maps_masked_pca = None

    # --- galactic mask / footprint ---
    if mask_galactic_plane == 1:
        galmask = np.load(FOOTPRINTS_DIR + f'footprint_intermediate_plus_Planck70_nside{nnside}_apodized.npy',
                          allow_pickle=True)
    else:
        galmask = np.ones(obs_maps.shape[1], dtype=bool)
    lmax = 3 * nnside

    # --- harmonic transforms ---
    print('Transforming to spherical harmonics ...')
    HIlm = hpyt.map2alm_parallel(hi_signal_smoothed, lmax=lmax, max_workers=20)
    Xlm = hpyt.map2alm_parallel(obs_maps, lmax=lmax, max_workers=20)
    HI_lup = lup_limit_data(HIlm, lup, lmax, nside=nnside)
    if bright_mask:
        Xlm_masked = hpyt.map2alm_parallel(obs_maps_masked, lmax=lmax, max_workers=20)
        Xlm_masked_pca = hpyt.map2alm_parallel(obs_maps_masked_pca, lmax=lmax, max_workers=20) \
            if mask_galactic_plane == 1 else None
    else:
        Xlm_masked = None
        Xlm_masked_pca = None

    # --- angular windows ---
    if window_mode == 'hard':
        win = wnd.make_hard_ell_windows(lmax, ell_edges=ell_edges, lup=lup)
        pca_window_ids = [0]
    elif window_mode == 'cosine':
        win = wnd.make_cosine_ell_windows(lmax=lmax, ell_edges=ell_edges, lup=lup, width=taper_width)
        pca_window_ids = [0]
    elif window_mode == 'wavelet':
        win = wnd.make_wavelet_ell_windows(lmax=lmax, n_wavelet_scales=n_wavelet_scales, lup=lup, renormalize=True)
        pca_window_ids = [n_wavelet_scales]     # coarse (last) scale cleaned with PCA
    else:
        raise ValueError("window_mode must be 'hard', 'cosine', or 'wavelet'")
    wnd.save_window_diagnostic_plot(win, results_output_path + f'ell_windows_{window_mode}.png')
    np.save(results_output_path + 'ell_windows.npy', win, allow_pickle=True)

    # =====================================================================
    #   Baseline: monoscale-JESSNet (no angular windows, no PCA block)
    # =====================================================================
    print('Running monoscale-JESSNet baseline (no windows, no PCA) ...')
    RecHI_monoscale, _, _ = run_monoscale_jessnet(obs_maps, obs_maps_masked, galmask, bl, ns_JESSNet,
                                                  K_max=0.9, c_wu=1e-2, nscales=4)

    # =====================================================================
    #   JESSNet: multiscale PCA + per-window cleaning
    # =====================================================================
    print(f'Running JESSNet multiscale cleaning ({window_mode}) ...')
    if window_mode == 'hard':
        Rlm = run_multiscale_hard(Xlm=Xlm, Xlm_masked=Xlm_masked, bl=bl, th=th, nnside=nnside,
                                  ell_edges=ell_edges, lup=lup, tau=tau, num_pca=ns_PCA, ns=ns_JESSNet,
                                  K_max=K_max, c_wu=c_wu, nscales=5, n_jobs_conv=5,
                                  overlap_sdec=overlap_sdec, galmask=galmask,
                                  Xlm_masked_pca=Xlm_masked_pca,
                                  dynamic_overlap=(mask_galactic_plane == 1))
    else:
        Rlm = run_multiscale_windowed(Xlm=Xlm, Xlm_masked=Xlm_masked, bl=bl, th=th, nnside=nnside,
                                      windows=win, pca_window_ids=pca_window_ids, tau=tau,
                                      channel_selection=channel_selection, num_pca=ns_PCA, ns=ns_JESSNet,
                                      K_max=K_max, c_wu=c_wu, nscales_sdec=5, n_jobs_conv=5,
                                      galmask=galmask, renormalize_available_windows=renormalize_available_windows,
                                      Xlm_masked_pca=Xlm_masked_pca)

    R = hpyt.alm2map(Rlm, nside=nnside)
    print('Cleaning done.')

    # =====================================================================
    #   Power-spectrum diagnostics
    # =====================================================================
    indexes_los = np.arange(hp.nside2npix(nnside))

    if mask_galactic_plane == 0:
        print('Computing full-sky power spectra ...')

        def _compute(nui):
            ell, cl_hi = spectra.Cell(HI_lup[nui, :])
            _, cl_jessnet = spectra.Cell(R[nui, :])
            _, cl_monoscale = spectra.Cell(RecHI_monoscale[nui, :])
            return cl_hi, cl_jessnet, cl_monoscale

        res = Parallel(n_jobs=10, prefer="processes")(delayed(_compute)(i) for i in range(freqs.shape[0]))
        cl_input = np.stack([r[0] for r in res], axis=0)
        cl_jessnet = np.stack([r[1] for r in res], axis=0)
        cl_monoscale = np.stack([r[2] for r in res], axis=0)
        ell, _ = spectra.Cell(HI_lup[0, :])

        k_input, P_input = spectra.plot_nuPk(HI_lup, indexes_los, freqs)
        k_jessnet, P_jessnet = spectra.plot_nuPk(R, indexes_los, freqs)
        k_monoscale, P_monoscale = spectra.plot_nuPk(RecHI_monoscale, indexes_los, freqs)

        freq_specs = {'JESSNet': (k_jessnet, P_jessnet), 'monoscaleJESSNet': (k_monoscale, P_monoscale),
                      'HI': (k_input, P_input)}
        ang_specs = {'ells': ell, 'JESSNet': cl_jessnet, 'monoscaleJESSNet': cl_monoscale, 'HI': cl_input}
        for name, (k, P) in freq_specs.items():
            np.save(results_output_path + f'freq_powspec_{name}.npy', np.stack((k, P), axis=0), allow_pickle=True)
        for key, value in ang_specs.items():
            np.save(results_output_path + f'angular_powspec_{key}.npy', value, allow_pickle=True)

    else:
        print('Computing footprint (masked) power spectra ...')
        mask_dict = np.load(FOOTPRINTS_DIR + f'footprint_intermediate_plus_Planck70_nside{nnside}_apodized_shrunk.npy',
                            allow_pickle=True).item()
        masks = [mask_dict[key] for key in mask_dict]
        names = list(mask_dict.keys())

        ki, Pi, elli, cli = spectra.post_GalMasking(masks, names, freqs, HI_lup, n_jobs=5)
        kj, Pj, ellj, clj = spectra.post_GalMasking(masks, names, freqs, R, n_jobs=5)
        km, Pm, ellm, clm = spectra.post_GalMasking(masks, names, freqs, RecHI_monoscale, n_jobs=5)

        freq_masked = {'HI': {'k': ki, 'P': Pi}, 'JESSNet': {'k': kj, 'P': Pj},
                       'monoscaleJESSNet': {'k': km, 'P': Pm}}
        ang_masked = {'HI': {'ell': elli, 'cl': cli}, 'JESSNet': {'ell': ellj, 'cl': clj},
                      'monoscaleJESSNet': {'ell': ellm, 'cl': clm}}
        for name, d in freq_masked.items():
            np.save(results_output_path + f'freq_powspec_GalMasked_{name}.npy', d, allow_pickle=True)
        for name, d in ang_masked.items():
            np.save(results_output_path + f'angular_powspec_GalMasked_{name}.npy', d, allow_pickle=True)

    print(f'Finished in {(time.time()-start)/60:.1f} min. Results in {results_output_path}')


if __name__ == '__main__':
    main()
