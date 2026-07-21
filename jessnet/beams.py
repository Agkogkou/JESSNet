"""Beam transfer functions.

Gaussian (optionally frequency-oscillating or degraded-to-common-resolution)
beam models in harmonic space. The ModAiry beam of the original code, which
required an external per-instrument beam file, is not included here.
"""

import numpy as np

C_LIGHT = 3.0e8  # m/s


def theta_FWHM(nu, dish_diam):
    """Gaussian-beam FWHM (radians) for frequency nu [MHz] and dish diameter [m]."""
    return C_LIGHT * 1e-6 / nu / float(dish_diam)


def getBeam(theta_fwhm, lmax):
    """Harmonic transfer function B_l of a Gaussian beam of given FWHM [rad]."""
    sigma_b = theta_fwhm / np.sqrt(8. * np.log(2.))
    l = np.linspace(0, lmax, lmax + 1)
    ell = l * (l + 1)
    return np.exp(-ell * sigma_b * sigma_b / 2)


def gen_beam_model(degraded, oscillating, freqs, nside, dish_diam=13.5,
                   A_beam=0.5, T_beam=20, nu0=None, no_beam=False):
    """Build the per-channel beam model.

    Returns
    -------
    th : (n_freq,) array
        Per-channel FWHM [rad].
    ell : (lmax+1,) array
        Multipole grid.
    beam_model : (n_freq, lmax+1) array
        Per-channel harmonic beam transfer function B_l.
    """
    print('Computing the beam ...')
    lmax = 3 * nside

    if no_beam:
        print(' No beam ...')
        return (np.zeros_like(freqs), np.linspace(0, lmax, lmax + 1),
                np.ones((freqs.shape[0], lmax + 1)))

    if degraded and oscillating:
        raise ValueError('degraded and oscillating beams cannot be combined.')

    th = theta_FWHM(freqs, dish_diam)

    if degraded:
        print(' Degraded (common-resolution) beam ...')
        th = np.ones_like(freqs) * th.max()
    elif oscillating:
        print(' Oscillating beam ...')
        if nu0 is None:
            nu0 = 0.
        th = th + np.deg2rad(A_beam / 60.0) * np.sin(2.0 * np.pi * (freqs - nu0) / T_beam)
    else:
        print(' Evolving (per-channel) beam ...')

    beam_model = np.array([getBeam(theta, lmax) for theta in th])
    return th, np.linspace(0, lmax, lmax + 1), beam_model
