"""Spherical-harmonic transforms and isotropic (starlet-like) spherical wavelets.

Thin wrappers around healpy plus the wavelet-filter machinery used by JESSNet.
Only the routines actually used by the pipeline are kept here.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import healpy as hp


# --------------------------------------------------------------------------
# Spherical harmonic transforms
# --------------------------------------------------------------------------
def _map2alm_one(args):
    m, lmax, niter = args
    os.environ["OMP_NUM_THREADS"] = "1"
    return hp.sphtfunc.map2alm(m, lmax=lmax, iter=niter)


def _alm2map_one(args):
    alm, nside = args
    os.environ["OMP_NUM_THREADS"] = "1"
    return hp.sphtfunc.alm2map(alm, nside)


def _smoothalm_one(args):
    alm, fl = args
    os.environ["OMP_NUM_THREADS"] = "1"
    return hp.sphtfunc.smoothalm(alm, beam_window=fl, inplace=False)


# --------------------------------------------------------------------------
# Persistent process pool for the small (n_sources-wide) per-source SHT
# stacks used inside the JESSNet solver's hot loop (wt_trans and the per-
# iteration Slm updates in core.py). healpy's map2alm/alm2map/smoothalm hold
# the GIL during their C computation, so a ThreadPoolExecutor gives *no*
# speedup there (measured: threaded ~= serial, slightly worse once thread
# overhead is counted). A process pool gives close to linear speedup instead
# (measured ~4.6x for n=5). The pool is created once and kept alive for the
# life of the process, so the one-time spawn cost is amortized across the
# ~100+ solver iterations that reuse it, instead of being paid every call.
#
# Deliberately uses the 'spawn' start method rather than the platform default
# ('fork' on Linux): this pool is first created *after* CUDA is already
# initialized in the main process (the learnlet model is moved onto the GPU
# in JESSNet.__init__, before the solver's first iteration), and forking a
# process that already holds a CUDA context is a known source of hangs/
# crashes in child processes. These workers never touch torch/CUDA, but
# spawning a clean interpreter sidesteps the issue entirely at negligible
# amortized cost.
# --------------------------------------------------------------------------
_FAST_SHT_POOL = None
_FAST_SHT_POOL_SIZE = 0
_FAST_SHT_CTX = mp.get_context("spawn")


def _fast_sht_pool(n):
    global _FAST_SHT_POOL, _FAST_SHT_POOL_SIZE
    if _FAST_SHT_POOL is None or _FAST_SHT_POOL_SIZE < n:
        if _FAST_SHT_POOL is not None:
            _FAST_SHT_POOL.shutdown(wait=True)
        _FAST_SHT_POOL = ProcessPoolExecutor(max_workers=n, mp_context=_FAST_SHT_CTX)
        _FAST_SHT_POOL_SIZE = n
    return _FAST_SHT_POOL


def map2alm_fast(maps, lmax, iter=3):
    """map2alm over a stack (n, p) of n>1 maps, process-parallel across n."""
    if maps.shape[0] == 1:
        return hp.sphtfunc.map2alm(maps[0], lmax=lmax, iter=iter)[None, :]
    pool = _fast_sht_pool(maps.shape[0])
    out = list(pool.map(_map2alm_one, [(maps[i], lmax, iter) for i in range(maps.shape[0])]))
    return np.array(out)


def alm2map_fast(alms, nside):
    """alm2map over a stack (n, t) of n>1 alms, process-parallel across n."""
    if alms.shape[0] == 1:
        return hp.sphtfunc.alm2map(alms[0], nside)[None, :]
    pool = _fast_sht_pool(alms.shape[0])
    out = list(pool.map(_alm2map_one, [(alms[i], nside) for i in range(alms.shape[0])]))
    return np.array(out)


def alm_product_fast(alms, filters):
    """Isotropic-filter product over a stack (n, t) of n>1 alms, one shared
    (t,) filter or one (n, t) filter per source, process-parallel across n."""
    n = alms.shape[0]
    if n == 1:
        fl = filters if len(np.shape(filters)) == 1 else filters[0, :]
        return hp.sphtfunc.smoothalm(alms[0, :], beam_window=fl, inplace=False)[None, :]
    pool = _fast_sht_pool(n)
    if len(np.shape(filters)) == 1:
        args = [(alms[i, :], filters) for i in range(n)]
    else:
        args = [(alms[i, :], filters[i, :]) for i in range(n)]
    out = list(pool.map(_smoothalm_one, args))
    return np.array(out)


def map2alm_parallel(maps, lmax=None, iter=3, max_workers=None):
    """map2alm over a stack of maps, parallelized across maps (processes)."""
    maps = np.asarray(maps)
    if maps.ndim == 1:
        if lmax is None:
            lmax = 3 * hp.get_nside(maps)
        return hp.sphtfunc.map2alm(maps, lmax=lmax, iter=iter)
    if lmax is None:
        lmax = 3 * hp.get_nside(maps[0])
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        out = list(ex.map(_map2alm_one, [(maps[i], lmax, iter) for i in range(maps.shape[0])]))
    return np.asarray(out)


def alm2map_parallel(alms, nside, max_workers=None):
    """alm2map over a stack of alms, parallelized across maps (processes)."""
    alms = np.asarray(alms)
    if alms.ndim == 1:
        return hp.sphtfunc.alm2map(alms, nside)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        out = list(ex.map(_alm2map_one, [(alms[i], nside) for i in range(alms.shape[0])]))
    return np.asarray(out)


def map2alm(maps, lmax=None, iter=3):
    """map2alm for a single map (p,) or a stack (n, p) -> (t,) or (n, t)."""
    if len(np.shape(maps)) == 1:
        if lmax is None:
            lmax = 3 * hp.get_nside(maps)
        return hp.sphtfunc.map2alm(maps, lmax=lmax, iter=iter)
    n = np.shape(maps)[0]
    if lmax is None:
        lmax = 3 * hp.get_nside(maps[0, :])
    return np.array([hp.sphtfunc.map2alm(maps[i, :], lmax=lmax, iter=iter) for i in range(n)])


def alm2map(alms, nside):
    """alm2map for a single alm (t,) or a stack (n, t) -> (p,) or (n, p)."""
    if len(np.shape(alms)) == 1:
        return hp.alm2map(alms, nside)
    n = np.shape(alms)[0]
    return np.array([hp.sphtfunc.alm2map(alms[i, :], nside) for i in range(n)])


def alm_product(alms, filters):
    """Apply an isotropic filter (lmax+1,) [or per-source (n, lmax+1)] to alm(s)."""
    dim_filters = len(np.shape(filters))
    dim_alms = len(np.shape(alms))
    if dim_filters == 1 and dim_alms == 1:
        return hp.sphtfunc.smoothalm(alms, beam_window=filters, inplace=False)
    n = np.shape(alms)[0]
    if dim_filters == 1:
        return np.array([hp.sphtfunc.smoothalm(alms[i, :], beam_window=filters, inplace=False)
                         for i in range(n)])
    return np.array([hp.sphtfunc.smoothalm(alms[i, :], beam_window=filters[i, :], inplace=False)
                     for i in range(n)])


def convolve(maps, filters, lmax=None, nside=None):
    """Convolve maps with isotropic harmonic filters (map -> alm -> filter -> map)."""
    if lmax is not None:
        if len(np.shape(filters)) == 1:
            lmax = len(filters) - 1
        else:
            lmax = np.shape(filters)[1] - 1
    alms = map2alm(maps, lmax=lmax)
    alms = alm_product(alms, filters)
    if nside is None:
        nside = hp.get_nside(maps)
    return alm2map(alms, nside=nside)


def convolve_parallel(maps_in, thetas, lmax, n_jobs=4):
    """Gaussian-smooth each map by its own FWHM `thetas[i]`, parallelized (processes)."""
    from joblib import Parallel, delayed
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(hp.smoothing)(maps_in[i], fwhm=thetas[i], lmax=lmax)
        for i in range(maps_in.shape[0])))


def anafast(maps, lmax=None, iter=3):
    """Angular power spectrum of a map (p,) or a stack (n, p)."""
    if len(np.shape(maps)) == 1:
        if lmax is None:
            lmax = 3 * hp.get_nside(maps)
        return hp.sphtfunc.anafast(maps, lmax=lmax, iter=iter)
    n = np.shape(maps)[0]
    if lmax is None:
        lmax = 3 * hp.get_nside(maps[0, :])
    return np.array([hp.sphtfunc.anafast(maps[i, :], lmax=lmax) for i in range(n)])


def alm2cl(alms):
    """Angular power spectrum from alm (t,) or stack (n, t)."""
    if len(np.shape(alms)) == 1:
        return hp.sphtfunc.alm2cl(alms)
    n = np.shape(alms)[0]
    return np.array([hp.sphtfunc.alm2cl(alms[i, :]) for i in range(n)])


# --------------------------------------------------------------------------
# alm index helpers
# --------------------------------------------------------------------------
def getsize(lmax):
    return hp.Alm.getsize(lmax)


def getlm(lmax):
    return hp.Alm.getlm(lmax)


def npix2nside(npix):
    return hp.npix2nside(npix)


# --------------------------------------------------------------------------
# Isotropic spherical-wavelet (starlet-like) filters
# --------------------------------------------------------------------------
def spline2(size, l, lc):
    """Non-negative decreasing B3-spline profile, value 1 at index 0."""
    res = np.arange(0, size + 1)
    res = 2 * l * res / (lc * size)
    res = (3 / 2) * 1 / 12 * (abs(res - 2) ** 3 - 4 * abs(res - 1) ** 3 + 6 * abs(res) ** 3
                              - 4 * abs(res + 1) ** 3 + abs(res + 2) ** 3)
    return res


def compute_h(size, lc):
    """Low-pass wavelet filter."""
    tab1 = spline2(size, 2 * lc, 1)
    tab2 = spline2(size, lc, 1)
    h = tab1 / (tab2 + 1e-6)
    h[int(size / (2 * lc)):size] = 0
    return h


def get_wt_filters(lmax, nscales):
    """Wavelet band filters, shape (lmax+1, nscales+1); last column is the coarse scale."""
    wt_filters = np.ones((lmax + 1, nscales + 1))
    wt_filters[:, 1:] = np.array([compute_h(lmax, 2 ** scale) for scale in range(nscales)]).T
    wt_filters[:, :nscales] -= wt_filters[:, 1:(nscales + 1)]
    return wt_filters


def wt_trans(inputs, nscales=3, lmax=None, alm_in=False, nside=None, alm_out=False):
    """Isotropic spherical wavelet transform -> (..., nscales+1) scale stack."""
    dim_inputs = len(np.shape(inputs))
    maps = None

    # For a stack of more than one source, every scale/source's SHT is fully
    # independent, so use the process-parallel variants (see their docstrings
    # for why threads don't help here). A single map/source falls back to the
    # plain serial calls, matching the pre-existing behavior exactly.
    _map2alm = map2alm_fast if dim_inputs > 1 else map2alm
    _alm2map = alm2map_fast if dim_inputs > 1 else alm2map
    _alm_product = alm_product_fast if dim_inputs > 1 else alm_product

    if alm_in:
        alms = inputs
        if nside is None and not alm_out:
            raise ValueError("nside is missing")
        if not alm_out:
            maps = _alm2map(alms, nside)
        if lmax is None:
            lmax = hp.Alm.getlmax(np.shape(alms)[-1])
    else:
        maps = inputs
        if dim_inputs == 1:
            nside = hp.get_nside(maps)
        else:
            nside = hp.get_nside(maps[0, :])
        if lmax is None:
            lmax = 3 * nside
        alms = _map2alm(maps, lmax=lmax)

    if not alm_out:
        l_scale = maps.copy()
        if dim_inputs == 1:
            npix = len(maps)
            wts = np.zeros((npix, nscales + 1))
        else:
            npix = np.shape(maps)[1]
            wts = np.zeros((np.shape(maps)[0], npix, nscales + 1))
    else:
        l_scale = alms.copy()
        if dim_inputs == 1:
            npix = np.size(alms)
            wts = np.zeros((npix, nscales + 1), dtype='complex')
        else:
            npix = np.shape(alms)[1]
            wts = np.zeros((np.shape(maps)[0], npix, nscales + 1), dtype='complex')

    scale = 1
    for j in range(nscales):
        h = compute_h(lmax, scale)
        if not alm_out:
            m = _alm2map(_alm_product(alms, h), nside)
        else:
            m = _alm_product(alms, h)
        h_scale = l_scale - m
        l_scale = m
        if dim_inputs == 1:
            wts[:, j] = h_scale
        else:
            wts[:, :, j] = h_scale
        scale *= 2

    if dim_inputs == 1:
        wts[:, nscales] = l_scale
    else:
        wts[:, :, nscales] = l_scale
    return wts


def wt_rec(wts):
    """Reconstruct a map from its wavelet scales (sum over the last axis)."""
    return np.sum(wts, axis=-1)
