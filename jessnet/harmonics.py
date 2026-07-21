"""Spherical-harmonic transforms and isotropic (starlet-like) spherical wavelets.

Thin wrappers around healpy plus the wavelet-filter machinery used by JESSNet.
Only the routines actually used by the pipeline are kept here.
"""

import os
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

    if alm_in:
        alms = inputs
        if nside is None and not alm_out:
            raise ValueError("nside is missing")
        if not alm_out:
            maps = alm2map(alms, nside)
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
        alms = map2alm(maps, lmax=lmax)

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
            m = alm2map(alm_product(alms, h), nside)
        else:
            m = alm_product(alms, h)
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
