"""End-to-end smoke test: does the JESSNet solver run without crashing?

This is not a correctness/recovery test (the "sources" here are just random
noise, not realistic foregrounds) -- it only checks that the beam-aware
source/mixing updates and the spherical-wavelet learnlet regularization run
on a tiny synthetic problem and produce finite, correctly-shaped output.

Requires the real project dependencies (numpy, healpy, torch, numba, scipy)
to be installed; skips itself otherwise. Runs on CPU (no GPU required).
"""

import numpy as np
import pytest

hp = pytest.importorskip("healpy")
torch = pytest.importorskip("torch")
pytest.importorskip("numba")
pytest.importorskip("scipy")

from jessnet import beams
from jessnet.core import JESSNet
from jessnet.learnlet import Learnlet


def _write_untrained_learnlet_weights(path, n_scales, kernel_size, filters):
    """A random-init Learnlet checkpoint, just so JESSNet has a weight file to
    load -- we're only testing that the pipeline runs, not reconstruction
    quality, so untrained weights are fine."""
    net = Learnlet(n_scales=n_scales, kernel_size=kernel_size, filters=filters,
                   exact_rec=True, thresh='hard', pretrained=False, device='cpu')
    torch.save({'state_dict': net.state_dict()}, path)


def test_jessnet_runs_on_tiny_synthetic_problem(tmp_path):
    rng = np.random.default_rng(0)

    nside = 4          # tiny map: 192 pixels
    nside_cut = 1       # -> 12 coarse cells of 4x4 pixels each (matches nside=4)
    n_freq = 5
    n_sources = 2
    learnlet_nscales = 3
    kernel_size = 3
    filters = 8

    freqs = np.linspace(900.0, 940.0, n_freq)
    _th, _ell, bl = beams.gen_beam_model(degraded=False, oscillating=False,
                                         freqs=freqs, nside=nside)

    npix = hp.nside2npix(nside)
    X = rng.normal(size=(n_freq, npix))
    galmask = np.ones(npix, dtype=bool)

    weights_path = tmp_path / "learnlet_untrained.pth"
    _write_untrained_learnlet_weights(str(weights_path), learnlet_nscales, kernel_size, filters)

    solver = JESSNet(
        X, bl, n_sources, galmask,
        perscale=True, minWuIt=2, c_wu=1e-2, c_ref=1e-2, cwuDec=2,
        nStd=0.22, useMad=True, nscales=2, k=3, K_max=0.6, L1=True, doRw=True,
        eps=np.array([1e-2, 1e-6, 1e-4]), verb=0, thrEnd=False, nnegA=False, nnegS=False,
        keepWuRegStr=False, cstWuRegStr=False, alm_in=False, device='cpu',
        learnlet_weight_path=str(weights_path), learnlet_nscales=learnlet_nscales,
        learnlet_nside_cut=nside_cut, learnlet_kernel_size=kernel_size,
        learnlet_filters=filters, learnlet_k=0.5, learnlet_batch_size=64,
    )

    status = solver.run()

    assert status == 0
    assert not solver.aborted
    assert solver.S.shape == (n_sources, npix)
    assert solver.A.shape == (n_freq, n_sources)
    assert np.all(np.isfinite(solver.S))
    assert np.all(np.isfinite(solver.A))
