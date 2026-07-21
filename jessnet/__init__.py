"""JESSNet: Joint dEconvolution and Sparse Separation Network.

A beam-aware, multiscale foreground-cleaning framework for 21 cm intensity
mapping, built on (S)DecGMCA with a spherical-wavelet learnlet as the learned
sparse regularization operator.
"""

from . import beams, harmonics, patches, preprocessing, spectra, windows, utils
from .core import JESSNet
from .learnlet import Learnlet
from .pipeline import (run_jessnet, run_monoscale_jessnet, run_PCA_temp,
                       run_multiscale_hard, run_multiscale_windowed, lup_limit_data)

__all__ = [
    "JESSNet", "Learnlet",
    "run_jessnet", "run_monoscale_jessnet", "run_PCA_temp",
    "run_multiscale_hard", "run_multiscale_windowed", "lup_limit_data",
    "beams", "harmonics", "patches", "preprocessing", "spectra", "windows", "utils",
]

__version__ = "0.1.0"
