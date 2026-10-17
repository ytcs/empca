"""empca -- optimal pulse processing with shift-invariant whitened PCA manifolds.

Stages
------
A. :class:`ShiftInvariantPCA`   -- whitened, shift-invariant EM-PCA (timing E-step,
   counter-rotated SVD M-step).
B. :class:`PulseManifold`       -- spline manifold through the origin in arc length.
C. :class:`ManifoldFitter`      -- Woodbury-accelerated non-stationary ML fit,
   with :class:`LowRankNoise` for ``R(E, t0) = R0 + T_t0 V Sigma(E) V^T T_t0^T``.

:class:`PulseProcessor` chains the three stages.  The frequency-domain EM-PCA
of Yu (2019) remains available as :class:`EMPCA` with :func:`ti_rfft` /
:func:`ti_irfft`.
"""
from ._legacy import EMPCA, ti_rfft, ti_irfft
from .calibration import EnergyCalibration, find_clusters, find_lines, line_resolution
from .covariance import LowRankNoise
from .filters import OptimalFilter
from .fitting import ManifoldFitter
from .manifold import PulseManifold
from .noise import StationaryNoise
from .pipeline import PulseFit, PulseProcessor
from .shifts import shift
from .sipca import ShiftInvariantPCA
from .timing import PhaseEqualizer

__version__ = "1.0.0"

__all__ = [
    "PulseProcessor", "PulseFit", "ShiftInvariantPCA", "PulseManifold", "ManifoldFitter",
    "LowRankNoise", "StationaryNoise", "OptimalFilter", "EnergyCalibration",
    "find_lines", "find_clusters", "line_resolution", "shift", "PhaseEqualizer",
    "EMPCA", "ti_rfft", "ti_irfft",
]
