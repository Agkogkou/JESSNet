"""Small numerical utilities: robust noise estimate and (optional) ground-truth
evaluation of a blind source separation solution."""

import numpy as np

from . import harmonics as hpyt


def mad(X=0, M=None):
    """Median-absolute-deviation robust std estimate (median|X-med| / 0.6735)."""
    if M is None:
        return np.median(abs(X - np.median(X))) / 0.6735
    xm = X[M == 1]
    return np.median(abs(xm - np.median(xm))) / 0.6735


def corr_perm(A0, S0, A, S, inplace=False, optInd=False):
    """Correct the source/column permutation and sign of a BSS solution against
    the ground truth (A0, S0)."""
    A0 = A0.copy()
    S0 = S0.copy()
    if not inplace:
        A = A.copy()
        S = S.copy()

    n = np.shape(A0)[1]

    for i in range(0, n):
        S[i, :] *= (1e-24 + np.linalg.norm(A[:, i]))
        A[:, i] /= (1e-24 + np.linalg.norm(A[:, i]))
        S0[i, :] *= (1e-24 + np.linalg.norm(A0[:, i]))
        A0[:, i] /= (1e-24 + np.linalg.norm(A0[:, i]))

    try:
        diff = abs(np.dot(np.linalg.inv(np.dot(A0.T, A0)), np.dot(A0.T, A)))
    except np.linalg.LinAlgError:
        diff = abs(np.dot(np.linalg.pinv(A0), A))
        print('Warning! Pseudo-inverse used.')

    ind = np.arange(0, n)
    for i in range(0, n):
        ind[i] = np.where(diff[i, :] == max(diff[i, :]))[0][0]

    A[:] = A[:, ind.astype(int)]
    S[:] = S[ind.astype(int), :]

    for i in range(0, n):
        p = np.sum(S[i, :] * S0[i, :])
        if p < 0:
            S[i, :] = -S[i, :]
            A[:, i] = -A[:, i]

    if inplace and not optInd:
        return None
    elif inplace and optInd:
        return ind
    elif not optInd:
        return A, S
    else:
        return A, S, ind


def nmse(S0, S):
    """Normalized mean square error (dB)."""
    return -10 * np.log10(np.sum((S0 - S) ** 2) / np.sum(S0 ** 2))


def ca(A0, A):
    """Criterion on the mixing matrix (dB)."""
    return -10 * np.log10(np.mean(np.abs(np.dot(np.linalg.pinv(A), A0) - np.eye(np.shape(A0)[1]))))


def evaluate(A0, S0, A, S, corrPerm=False, perScale=False, nscales=3, S0wt=None):
    """Compute CA and NMSE (optionally per wavelet scale) after permutation correction."""
    if not corrPerm:
        A = A.copy()
        S = S.copy()

    n = np.shape(A0)[1]
    corr_perm(A0, S0, A, S, inplace=True)

    CA = -10 * np.log10(np.mean(np.abs(np.dot(np.linalg.pinv(A), A0) - np.eye(n))))
    NMSE = -10 * np.log10(np.sum((S0 - S) ** 2) / np.sum(S0 ** 2))

    if not perScale:
        return CA, NMSE

    if S0wt is not None:
        nscales = np.shape(S0wt)[2] - 1
    else:
        S0wt = hpyt.wt_trans(S0, nscales=nscales)
    Swt = hpyt.wt_trans(S, nscales=nscales)
    NMSEScale = -10 * np.log10(np.sum((S0wt - Swt) ** 2, axis=(0, 1)) / np.sum(S0 ** 2, axis=(0, 1)))
    return CA, NMSE, NMSEScale
