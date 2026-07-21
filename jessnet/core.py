"""The JESSNet solver: beam-aware joint deconvolution + blind source separation
in spherical-harmonic space, with a spherical-wavelet learnlet as the learned
sparse regularization operator.

This is a single, speed-optimized class. The `perscale` flag selects the two
behaviours that differ between a run on the full data and a run on a single
angular window:
    perscale=False : masked alms always built; no oblique A normalization
    perscale=True  : masked alms only when a mask is present; oblique A normalization

Module-level configuration (set before running):
    PROFILE          -> print a per-section wall-clock breakdown after each run
    PARALLEL_SHT     -> transform the n sources concurrently (threads)
    WEIGHT_PATH,LEARNLET_* -> spherical-learnlet weights and hyperparameters
"""

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import healpy as hp
import torch

from . import harmonics as hpyt
from . import utils
from .patches import from_healpix_to_patches_numba, from_patches_to_healpix_numba
from .learnlet import Learnlet


# ---- run-time switches ----------------------------------------------------
PROFILE = False        # print a per-section timing breakdown after run()
PARALLEL_SHT = True    # transform the n sources concurrently (threads)


def _default_device():
    """cuda > mps (Apple Silicon) > cpu. The learnlet forward pass dominates
    JESSNet's runtime, so MPS is a large (~20x measured) speedup over CPU on
    Apple Silicon and is worth preferring over CPU when no CUDA GPU exists."""
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ---- spherical-learnlet configuration (override before a run) -------------
WEIGHT_PATH = 'weights/learnlet_sphere_64_5_sc5_fg.pth'
LEARNLET_NSCALES = 5        # total spherical-wavelet scales = 1 coarse + (L-1) detail; must match the weights
LEARNLET_NSIDE_CUT = 4      # HEALPix patching resolution
LEARNLET_K = 0.5           # noise level fed to the net: sigma = k * MAD(source)  (the tunable knob)
LEARNLET_BATCH_SIZE = 400
LEARNLET_KERNEL_SIZE = 5
LEARNLET_FILTERS = 64
LEARNLET_THRESH = 'hard'


# --------------------------------------------------------------------------
# Parallel spherical-harmonic transforms (threaded over sources; identical output)
# --------------------------------------------------------------------------
_SHT_POOL = None


def _sht_pool(n):
    global _SHT_POOL
    if _SHT_POOL is None or _SHT_POOL._max_workers < n:
        _SHT_POOL = ThreadPoolExecutor(max_workers=n)
    return _SHT_POOL


def map2alm_maybe_parallel(maps, lmax, niter):
    if maps.ndim == 1:
        return hp.map2alm(maps, lmax=lmax, iter=niter)
    if not PARALLEL_SHT or maps.shape[0] == 1:
        return np.array([hp.map2alm(maps[i], lmax=lmax, iter=niter) for i in range(maps.shape[0])])
    pool = _sht_pool(maps.shape[0])
    return np.array(list(pool.map(lambda mm: hp.map2alm(mm, lmax=lmax, iter=niter), maps)))


def alm2map_maybe_parallel(alms, nside):
    if alms.ndim == 1:
        return hp.alm2map(alms, nside)
    if not PARALLEL_SHT or alms.shape[0] == 1:
        return np.array([hp.alm2map(alms[i], nside) for i in range(alms.shape[0])])
    pool = _sht_pool(alms.shape[0])
    return np.array(list(pool.map(lambda aa: hp.alm2map(aa, nside), alms)))


# --------------------------------------------------------------------------
# Cached HEALPix <-> patch index tables (constant across iterations)
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _pixs_in(nside_map, nside_cut):
    ncoarse = hp.nside2npix(nside_cut)
    npix = hp.nside2npix(nside_map) // ncoarse
    pixs = np.array([np.sort(hp.nest2ring(nside_map, np.arange(i * npix, (i + 1) * npix, dtype=np.int64)))
                     for i in range(ncoarse)], dtype=np.int64)
    return pixs, ncoarse, int(np.sqrt(npix))


@lru_cache(maxsize=None)
def _pixs_out(nside_out, npix_patch, ncoarse):
    pp = npix_patch * npix_patch
    return np.array([np.sort(hp.nest2ring(nside_out, np.arange(i * pp, (i + 1) * pp, dtype=np.int64)))
                     for i in range(ncoarse)], dtype=np.int64)


def to_patches_cached(maps, nside_cut=4):
    nside_map = hp.npix2nside(maps.shape[1])
    pixs, ncoarse, npix_patch = _pixs_in(nside_map, nside_cut)
    return from_healpix_to_patches_numba(maps, pixs, ncoarse, npix_patch)


def to_healpix_cached(patches):
    ncoarse = patches.shape[1]
    npix_patch = patches.shape[2]
    nb_maps = patches.shape[0]
    nside_out = hp.npix2nside(ncoarse * npix_patch * npix_patch)
    pixs_out = _pixs_out(nside_out, npix_patch, ncoarse)
    return from_patches_to_healpix_numba(patches, pixs_out, nb_maps, hp.nside2npix(nside_out))


class JESSNet:
    """Joint dEconvolution and Sparse Separation Network solver for one dataset
    (either the full data or a single angular window)."""

    def __init__(self, X, Hl, n, galmask, **kwargs):

        self.perscale = kwargs.get('perscale', False)

        # given attributes
        self.Hl = Hl
        self.n = n
        self.M = kwargs.get('M', None)
        self.AInit = kwargs.get('AInit', None)
        nneg = kwargs.get('nneg', None)
        if nneg is not None:
            self.nnegA = nneg
            self.nnegS = nneg
        else:
            self.nnegA = kwargs.get('nnegA', False)
            self.nnegS = kwargs.get('nnegS', False)
        self.keepWuRegStr = kwargs.get('keepWuRegStr', False)
        self.cstWuRegStr = kwargs.get('cstWuRegStr', False)
        self.minWuIt = kwargs.get('minWuIt', 50)
        self.c_wu = kwargs.get('c_wu', 1e-2)
        self.c_ref = kwargs.get('c_ref', .5)
        self.cwuDec = kwargs.get('cwuDec', int(self.minWuIt / 2))
        self.useMad = kwargs.get('useMad', True)
        if 'nStd' not in kwargs and (not self.keepWuRegStr or not self.useMad):
            raise KeyError('nStd must be provided for spectrum-based regularization or noise std calc.')
        else:
            self.nStd = kwargs.get('nStd', 0.)
        self.nscales = kwargs.get('nscales', 3)
        self.k = kwargs.get('k', 3)
        self.K_max = kwargs.get('K_max', .5)
        self.L1 = kwargs.get('L1', True)
        self.doRw = kwargs.get('doRw', True)
        self.thrEnd = kwargs.get('thrEnd', True)
        self.iterSH = kwargs.get('iterSH', 3)
        self.eps = kwargs.get('eps', np.array([1e-2, 1e-4, 1e-4]))
        self.verb = kwargs.get('verb', 0)
        self.S0 = kwargs.get('S0', None)
        self.A0 = kwargs.get('A0', None)
        self.iSNR0 = kwargs.get('iSNR0', None)
        self.epsilon = []

        # deduced attributes
        self.m = np.shape(X)[0]
        alm_in = kwargs.get('alm_in', False)
        if not alm_in:
            self.p = np.shape(X)[1]
        else:
            self.p = int(12 * (2 ** int(np.log2((-3 + np.sqrt(9 + 8 * (np.shape(X)[1] - 1))) / 6))) ** 2)
        self.supp = int(np.sum(self.M)) if self.M is not None else self.p

        self.lmax = np.shape(self.Hl)[1] - 1
        self.t = hpyt.getsize(self.lmax)
        self.nside = hpyt.npix2nside(self.p)
        self.ls, self.ms = hpyt.getlm(self.lmax)
        self.factors = 2 - (self.ms == 0)
        self.nms = np.array([2 * l + 1 for l in range(self.lmax + 1)])
        self.Hlm = np.concatenate([Hl[:, l:] for l in range(self.lmax + 1)], axis=1)
        self.wt_filters = hpyt.get_wt_filters(lmax=self.lmax, nscales=self.nscales)
        self.galmask = galmask

        if not alm_in:
            self.Xlm = hpyt.map2alm(X, lmax=self.lmax, iter=self.iterSH)
        else:
            self.Xlm = X

        # masked alms (detail scales of the data used in the A update)
        if self.perscale:
            if not np.all(self.galmask == 1):
                X_map = hpyt.alm2map(X, nside=self.nside) if alm_in else X
                X_masked = np.copy(X_map)
                for i in range(len(X_map)):
                    X_masked[i] = X_map[i] * galmask
                self.Xlm_masked = hpyt.map2alm(X_masked, lmax=self.lmax, iter=self.iterSH)
                self.Xlm_det = hpyt.alm_product(self.Xlm_masked, 1 - self.wt_filters[:, -1])
            else:
                self.Xlm_det = hpyt.alm_product(self.Xlm, 1 - self.wt_filters[:, -1])
        else:
            X_masked = np.copy(X)
            for i in range(len(X)):
                X_masked[i] = X[i] * galmask
            self.Xlm_masked = hpyt.map2alm(X_masked, lmax=self.lmax, iter=self.iterSH)
            self.Xlm_det = hpyt.alm_product(self.Xlm_masked, 1 - self.wt_filters[:, -1])

        self.nStdSH = self.nStd * np.sqrt(4 * np.pi / self.p)
        if self.S0 is not None and self.iSNR0 is None:
            self.iSNR0 = self.nStdSH ** 2 / hpyt.anafast(self.S0, lmax=self.lmax) * self.supp / self.p
        if self.S0 is not None:
            self.S0wt = hpyt.wt_trans(self.S0, nscales=self.nscales)

        # state
        self.S = np.zeros((self.n, self.p))
        self.Slm = np.zeros((self.n, self.t), dtype=complex)
        self.Slm_det = np.zeros((self.n, self.t), dtype=complex)
        self.A = np.zeros((self.m, self.n))
        self.invOpSp = np.zeros((self.n, self.lmax + 1))
        self.lastWuIt = None
        self.lastRefIt = None
        self.nmse = None
        self.ca = None
        self.nmseScales = None
        self.aborted = False
        self.device = torch.device(kwargs.get('device', _default_device()))

        # caches (built lazily)
        self._Hl2_cache = None
        self._factors_Xlm_det_Hlm_cache = None
        self._ell_sort_cache = None
        self._ell_starts_cache = None
        self._Xs_cache = None
        self._fac_sorted_cache = None

        # profiling
        self.profile = kwargs.get('profile', PROFILE)
        self._timings = defaultdict(float)
        self._counts = defaultdict(int)

        # spherical learnlet
        self._setup_learnlet(kwargs)

    # ----------------------------------------------------------------------
    def _setup_learnlet(self, kwargs):
        """Load the spherical-wavelet learnlet and its weights."""
        self.learnlet_nscales = kwargs.get('learnlet_nscales', LEARNLET_NSCALES)
        self.learnlet_nside_cut = kwargs.get('learnlet_nside_cut', LEARNLET_NSIDE_CUT)
        self.learnlet_k = float(kwargs.get('learnlet_k', LEARNLET_K))
        self.learnlet_batch_size = kwargs.get('learnlet_batch_size', LEARNLET_BATCH_SIZE)
        self.learnlet_weight_path = kwargs.get('learnlet_weight_path', WEIGHT_PATH)

        if self.learnlet_nscales < 2:
            raise ValueError('learnlet_nscales must be >= 2.')

        self.learnlet = Learnlet(
            n_scales=self.learnlet_nscales,
            kernel_size=kwargs.get('learnlet_kernel_size', LEARNLET_KERNEL_SIZE),
            filters=kwargs.get('learnlet_filters', LEARNLET_FILTERS),
            exact_rec=True,
            thresh=kwargs.get('learnlet_thresh', LEARNLET_THRESH),
            pretrained=False,
            device=str(self.device),
        ).to(self.device)

        ckpt = torch.load(self.learnlet_weight_path, map_location=self.device, weights_only=False)
        state_dict = ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else ckpt
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.learnlet.load_state_dict(state_dict, strict=True)
        self.learnlet.eval()
        print(f'[learnlet] loaded {self.learnlet_weight_path} '
              f'(n_scales={self.learnlet_nscales}, k={self.learnlet_k})')

    # ----------------------------------------------------------------------
    def run(self):
        """Run the full JESSNet separation (warm-up, refinement, final refinement)."""
        self.initialize()
        if self.core() == 1:
            self.aborted = True
            return 1
        refine_s_end, epsilon = self.refine_s_end()
        self.epsilon.append(epsilon)
        if refine_s_end == 1:
            self.aborted = True
            return 1
        self.terminate()
        if self.profile:
            self._print_timings()
        return 0

    def _print_timings(self):
        leaves = {k: v for k, v in self._timings.items() if k != 'update_s_total'}
        total = sum(leaves.values())
        print("\n  ---- JESSNet per-section timing (perscale=%s) ----" % self.perscale)
        print("  %-18s %10s %8s %12s" % ("section", "total[s]", "calls", "per-call[ms]"))
        for label, tval in sorted(leaves.items(), key=lambda kv: -kv[1]):
            c = self._counts.get(label, 0)
            percall = (tval / c * 1e3) if c else float('nan')
            frac = 100 * tval / total if total else 0
            print("  %-18s %10.1f %8d %12.2f  (%4.1f%%)" % (label, tval, c, percall, frac))
        print("  %-18s %10.1f" % ("SUM(leaves)", total))
        print("  ---------------------------------------------------\n")

    def initialize(self):
        """PCA initialization of the mixing matrix and reset of the state."""
        Xlm = hpyt.alm_product(self.Xlm, self.Hl[0, :] / (self.Hl + 1e-10))
        if self.AInit is not None:
            self.A = self.AInit.copy()
        else:
            R = np.real(Xlm @ Xlm.T.conj())
            _, V = np.linalg.eig(R)
            self.A = V[:, 0:self.n]
        self.A /= np.maximum(np.linalg.norm(self.A, axis=0), 1e-24)

        self.S = np.zeros((self.n, self.p))
        self.Slm = np.zeros((self.n, self.t), dtype=complex)
        self.lastWuIt = None
        self.lastRefIt = None
        self.aborted = False
        return 0

    def core(self):
        """Alternate S and A updates through the warm-up and refinement stages."""
        stage = "wu"
        S_old = np.zeros((self.n, self.p))
        A_old = np.zeros((self.m, self.n))
        it = 0

        while True:
            it += 1
            strat, c, K, doRw, nnegS = self.get_parameters(stage, it)
            strat = 1  # strategy #3 of Carloni Gertosio & Bobin (2021) for all stages

            t0 = time.perf_counter()
            update_s, epsilon = self.update_s(strat, c, K, it, doRw=doRw, nnegS=nnegS)
            self._timings['update_s_total'] += time.perf_counter() - t0
            self._counts['update_s_total'] += 1
            self.epsilon.append(epsilon)
            if update_s:
                return 1

            t0 = time.perf_counter()
            update_a = self.update_a()
            self._timings['update_a'] += time.perf_counter() - t0
            self._counts['update_a'] += 1
            if update_a:
                return 1

            maskk = self.galmask != 0
            delta_S = np.linalg.norm(S_old[:, maskk] - self.S[:, maskk]) / np.linalg.norm(self.S[:, maskk])
            S_old = self.S.copy()
            A_old = self.A.copy()

            if self.verb >= 2:
                cond_A = np.linalg.cond(self.A)
                print("it %i: delta_S = %.2e - cond(A) = %.2f" % (it, delta_S, cond_A))

            if stage == 'wu' and it >= self.minWuIt and (delta_S <= self.eps[0] or it >= self.minWuIt + 50):
                if self.verb >= 2:
                    print("> End of the warm-up (iteration %i)" % it)
                self.lastWuIt = it
                stage = 'ref'

            if stage == 'ref' and (delta_S <= self.eps[1] or it >= self.lastWuIt + 50) and (it >= self.lastWuIt + 25):
                if self.verb >= 2:
                    print("> End of the refinement (iteration %i)" % it)
                self.lastRefIt = it
                return 0

    def get_parameters(self, stage, it):
        """Stage/iteration-dependent regularization parameters."""
        strat = 0 if self.cstWuRegStr else 1
        if stage == 'wu':
            if np.isscalar(self.c_wu):
                c = self.c_wu
            else:
                c = np.maximum(np.min(self.c_wu),
                               np.max(self.c_wu) * 10 ** ((np.log10(np.min(self.c_wu)) - np.log10(np.max(self.c_wu)))
                                                          * (it - 1) / (self.cwuDec - 1)))
            K = np.minimum(self.K_max / self.minWuIt * it, self.K_max)
            doRw = False
            nnegS = False
        else:
            if self.keepWuRegStr:
                c = np.min(self.c_wu)
            else:
                strat = 2
                c = self.c_ref
            K = self.K_max
            doRw = self.doRw
            nnegS = self.nnegS
        return strat, c, K, doRw, nnegS

    def update_s(self, strat, c, K, iteration, doThr=True, doRw=None, nnegS=None, Slm=None, Slm_det=None, S=None,
                 A=None, iSNR=None, stds=None, Swtrw=None, oracle=False):
        """Least-square source update followed by learnlet regularization."""
        if nnegS is None:
            nnegS = self.nnegS
        if doRw is None:
            doRw = self.doRw

        t0 = time.perf_counter()
        ls_s, epsilon = self.ls_s(strat, c, iteration, Slm=Slm, A=A, iSNR=iSNR, oracle=oracle)
        self._timings['ls_s'] += time.perf_counter() - t0
        self._counts['ls_s'] += 1
        if ls_s:
            return 1

        self._threshold_wl1(doThr)
        return 0, epsilon

    def _threshold_wl1(self, doThr):
        """Spherical-wavelet learnlet denoising of the current sources.

        Mirrors the learnlet training pipeline: per-source unit-variance
        normalization -> spherical wavelet scale cube -> patches -> learnlet
        (with sigma = k * MAD) -> reassemble -> un-normalize + mask.
        """
        L = self.learnlet_nscales
        nside_cut = self.learnlet_nside_cut

        t0 = time.perf_counter()
        self.S = alm2map_maybe_parallel(self.Slm, self.nside)
        self._timings['alm2map'] += time.perf_counter() - t0
        self._counts['alm2map'] += 1

        if not doThr:
            return

        maskk = self.galmask != 0

        mu = np.array([self.S[c][maskk].mean() for c in range(self.n)])
        sd = np.maximum(np.array([self.S[c][maskk].std() for c in range(self.n)]), 1e-30)
        S_norm = (self.S - mu[:, None]) / sd[:, None]

        # per-source noise level fed to the net (the tunable knob)
        sig_src = np.array([self.learnlet_k * utils.mad(S_norm[c][maskk])
                            for c in range(self.n)], dtype=np.float32)

        t0 = time.perf_counter()
        Swt = hpyt.wt_trans(S_norm, nscales=L - 1, lmax=self.lmax, alm_in=False)  # (n, npix, L)
        self._timings['wt_trans'] += time.perf_counter() - t0
        self._counts['wt_trans'] += 1
        if Swt.shape != (self.n, self.p, L):
            raise ValueError(f'wt_trans gave {Swt.shape}, expected {(self.n, self.p, L)}; check learnlet_nscales.')
        S_cube = np.moveaxis(Swt, 2, 1)                                          # (n, L, npix)
        S_flat = np.ascontiguousarray(S_cube.reshape(self.n * L, self.p).astype(np.float32))

        t0 = time.perf_counter()
        patches = to_patches_cached(S_flat, nside_cut=nside_cut)                 # (n*L, n_patches, H, W)
        n_patches, H, W = patches.shape[1], patches.shape[2], patches.shape[3]
        patches = patches.reshape(self.n, L, n_patches, H, W).swapaxes(1, 2)     # (n, n_patches, L, H, W)
        self._timings['to_patches'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        out_maps = np.empty((self.n, n_patches, H, W), dtype=np.float32)
        with torch.no_grad():
            for comp in range(self.n):
                xt = torch.from_numpy(np.ascontiguousarray(patches[comp])).float().to(self.device)
                for start in range(0, n_patches, self.learnlet_batch_size):
                    stop = min(start + self.learnlet_batch_size, n_patches)
                    batch = xt[start:stop]                                       # (b, L, H, W)
                    sigma = torch.full((batch.shape[0],), float(sig_src[comp]),
                                       device=self.device, dtype=torch.float32)
                    out = self.learnlet(batch, sigma)                            # (b, 1, H, W)
                    out_maps[comp, start:stop] = out[:, 0].detach().cpu().numpy()
        self._timings['learnlet'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        S_clean_norm = to_healpix_cached(out_maps)
        self._timings['to_healpix'] += time.perf_counter() - t0

        self.S = (S_clean_norm * sd[:, None] + mu[:, None]) * self.galmask[None, :]

        t0 = time.perf_counter()
        self.Slm = map2alm_maybe_parallel(self.S, self.lmax, self.iterSH)
        self._timings['map2alm'] += time.perf_counter() - t0
        self._counts['map2alm'] += 1

        t0 = time.perf_counter()
        self.Slm_det = hpyt.alm_product(self.Slm, 1 - self.wt_filters[:, -1])
        self._timings['alm_product'] += time.perf_counter() - t0

    # ----------------------------------------------------------------------
    def _ensure_ell_cache(self):
        """Constant tables grouping alm coefficients by multipole (built once)."""
        if self._ell_sort_cache is None:
            self._ell_sort_cache = np.argsort(self.ls, kind='stable')
            counts = np.bincount(self.ls, minlength=self.lmax + 1)
            self._ell_starts_cache = np.concatenate(([0], np.cumsum(counts)))
            self._Xs_cache = np.ascontiguousarray(self.Xlm[:, self._ell_sort_cache])
            self._fac_sorted_cache = self.factors[self._ell_sort_cache]

    def ls_s(self, strat, c, iteration, Slm=None, A=None, iSNR=None, oracle=False):
        """Tikhonov-regularized beam-aware least-square source update (per mode)."""
        if Slm is None:
            Slm = self.Slm
        if A is None:
            A = self.A if not oracle else self.A0

        if strat == 2 and iSNR is None:
            if not oracle:
                spectra = hpyt.alm2cl(Slm)
                spectra = np.maximum(spectra, np.max(spectra, axis=1)[:, np.newaxis] * 1e-20)
                iSNR = self.nStdSH ** 2 / spectra * self.supp / self.p
            else:
                iSNR = self.iSNR0

        normAA = np.linalg.norm(A.T @ A, ord=-2)

        if self._Hl2_cache is None:
            self._Hl2_cache = self.Hl ** 2
        Hl2 = self._Hl2_cache

        Ra = np.einsum('lj,li,lk', A, Hl2, A, optimize=True)
        if strat == 0:
            Ra += c * np.eye(self.n)[np.newaxis, :, :]
            epsilon = c * np.eye(self.n)
        elif strat == 2:
            eps = np.zeros((self.lmax + 1, self.n, self.n))
            diag = np.arange(self.n)
            eps[:, diag, diag] = c * iSNR.T
            epsilon = c * iSNR.T
            Ra += eps
        else:  # mixing-matrix-based regularization (strategy #3)
            reg = np.maximum(0, c - np.linalg.norm(Ra, ord=-2, axis=(1, 2)) / normAA)
            epsilon = reg
            Ra += reg[:, np.newaxis, np.newaxis] * np.eye(self.n)[np.newaxis, :, :]

        try:
            Ua, Sa, Va = np.linalg.svd(Ra)
        except np.linalg.LinAlgError:
            if self.verb:
                print('SVD did not converge, abort')
            return 1
        Sa = np.maximum(Sa, np.max(Sa, axis=1)[:, np.newaxis] * 1e-9)
        iRa = np.einsum('...ki,...k,...jk', Va, 1 / Sa, Ua, optimize=True)
        piA = np.einsum('ijk,lk,li->ijl', iRa, A, self.Hl, optimize=True)         # (lmax+1, n, m)

        # Apply piA[ell] per mode: Slm[:, i] = piA[ls[i]] @ Xlm[:, i], grouped by
        # multipole so each block is a small contiguous BLAS matmul (no giant array).
        self._ensure_ell_cache()
        sort = self._ell_sort_cache
        starts = self._ell_starts_cache
        Xs = self._Xs_cache
        Ss = np.empty((self.n, self.t), dtype=complex)
        for l in range(self.lmax + 1):
            a, b = starts[l], starts[l + 1]
            Ss[:, a:b] = piA[l] @ Xs[:, a:b]
        Slm[:, sort] = Ss

        if not self.useMad:
            self.invOpSp = np.einsum('ijk,ijk->ji', piA, piA, optimize=True)

        return 0, epsilon

    def update_a(self, Slm_det=None, A=None):
        """Least-square mixing-matrix update using the detail scales of data/sources."""
        if Slm_det is None:
            Slm_det = self.Slm_det
        if A is None:
            A = self.A

        if self._factors_Xlm_det_Hlm_cache is None:
            self._factors_Xlm_det_Hlm_cache = self.factors * self.Xlm_det * self.Hlm
        if self._Hl2_cache is None:
            self._Hl2_cache = self.Hl ** 2
        fXdHlm = self._factors_Xlm_det_Hlm_cache

        # Rs[i,j,k] = sum_l factors[l] Hl[i,ell(l)]^2 Slm_det[j,l] conj(Slm_det[k,l]).
        # Hl is isotropic -> bin the alm cross-products by multipole first (t -> lmax+1),
        # then contract the small ell axis against Hl^2.
        self._ensure_ell_cache()
        sort = self._ell_sort_cache
        starts = self._ell_starts_cache
        Sd_s = Slm_det[:, sort]
        Sw = Sd_s * self._fac_sorted_cache
        Q = np.empty((self.lmax + 1, self.n, self.n), dtype=complex)
        for l in range(self.lmax + 1):
            a, b = starts[l], starts[l + 1]
            Q[l] = Sw[:, a:b] @ np.conj(Sd_s[:, a:b]).T
        Rs = np.real(self._Hl2_cache @ Q.reshape(self.lmax + 1, self.n * self.n)).reshape(self.m, self.n, self.n)

        try:
            Us, Ss, Vs = np.linalg.svd(Rs)
        except np.linalg.LinAlgError:
            if self.verb:
                print('SVD did not converge, abort')
            return 1
        Ss = np.maximum(Ss, np.max(Ss, axis=1)[:, np.newaxis] * 1e-9)
        iRs = np.einsum('...ij,...j,...jk', Us, 1 / Ss, Vs, optimize=True)
        Ws = np.real(np.einsum('ij,kj->ik', fXdHlm, np.conj(Slm_det), optimize=True))
        A[:] = np.einsum('ij,ijk->ik', Ws, iRs, optimize=True)

        if self.nnegA:
            sign = np.sign(np.sum(A, axis=0))
            sign[sign == 0] = 1
            A *= sign
            A[:] = np.maximum(A, 0)

        if self.perscale:  # oblique constraint
            A /= np.maximum(np.linalg.norm(A, axis=0), 1e-24)
        return 0

    def refine_s_end(self):
        """Final source-only refinement with the fixed final mixing matrix."""
        strat = 0 if self.cstWuRegStr else 1
        c = np.min(self.c_wu)

        update_s, epsilon = self.update_s(strat, c, 1, -1, doThr=self.thrEnd, doRw=False)
        if update_s:
            return 1, epsilon

        S_old = np.zeros((self.n, self.p))
        delta_S = np.inf
        it = 0
        if not self.keepWuRegStr:
            strat = 2
            c = self.c_ref

        while delta_S >= self.eps[2] and it < 25:
            it += 1
            update_s, epsilon = self.update_s(strat, c, 1, it, doThr=self.thrEnd)
            self.epsilon.append(epsilon)
            if update_s:
                return 1, epsilon
            maskk = self.galmask != 0
            delta_S = np.linalg.norm(S_old[:, maskk] - self.S[:, maskk]) / np.linalg.norm(self.S[:, maskk])
            S_old = self.S.copy()
            if self.verb >= 2:
                print("final refinement it %i: delta_S = %.2e" % (it, delta_S))
        return 0, epsilon

    def terminate(self):
        """If ground truth is provided, correct permutations and evaluate the solution."""
        if self.A0 is not None and self.S0 is not None:
            self.ca, self.nmse, self.nmseScales = utils.evaluate(
                self.A0, self.S0, self.A, self.S, corrPerm=True, perScale=True, S0wt=self.S0wt)
            if self.verb:
                print('CA : %.2f | NMSE: %.2f' % (self.ca, self.nmse))
        return 0
