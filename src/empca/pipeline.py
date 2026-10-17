"""High-level, scikit-learn style interface to the three-stage pipeline."""
from dataclasses import dataclass

import numpy as np

from .calibration import EnergyCalibration, find_clusters
from .covariance import LowRankNoise
from .fitting import ManifoldFitter
from .manifold import PulseManifold
from .noise import StationaryNoise
from .shifts import shift_rfft
from .sipca import ShiftInvariantPCA

__all__ = ["PulseProcessor", "PulseFit"]


@dataclass
class PulseFit:
    """Per-record results of :meth:`PulseProcessor.transform`."""

    s: np.ndarray        # manifold coordinate (whitened arc length = cumulative SNR)
    t0: np.ndarray       # arrival time [samples] in the learned gauge
    nll: np.ndarray      # -2 log-likelihood (up to a constant)
    coeffs: np.ndarray   # aligned whitened PCA coefficients (Stage A)
    energy: np.ndarray = None  # calibrated energy, if a calibration is set

    def __len__(self):
        return len(self.s)


class PulseProcessor:
    """Shift-invariant whitened PCA + spline manifold + Woodbury ML fit.

    Parameters
    ----------
    n_components : int
        Dimension of the shift-invariant pulse subspace (Stage A).
    n_align : int
        Leading components used for alignment (see :class:`ShiftInvariantPCA`).
    max_shift : int
        Arrival-time search window [samples].
    snr_min : float, optional
        Minimum whitened SNR of the records used to learn the basis.
    covariance : {"lines", "curve", None}
        Non-stationary noise model (Stage C).  ``"lines"`` estimates it from
        the mono-energetic clusters found in the data (no energy labels
        needed) and captures fluctuations along the manifold, ``"curve"`` uses
        residuals about the manifold (fully label-free, orthogonal
        fluctuations only), ``None`` gives the stationary ML fit.
    n_lines : int, optional
        Number of spectral lines to look for when ``covariance="lines"``.
    n_noise_modes : int
        Rank ``K_n`` of the non-stationary excess covariance.
    manifold : dict, optional
        Keyword arguments for :class:`PulseManifold`.

    Examples
    --------
    >>> proc = PulseProcessor(n_components=4, n_lines=3).fit(traces, noise_traces)
    >>> res = proc.transform(traces)
    >>> proc.calibrate(proc.line_centers_, [1500., 5900., 8000.])
    >>> energies = proc.energy(res.s)
    """

    def __init__(self, n_components=4, n_align=1, max_shift=16, snr_min=None,
                 covariance="lines", n_lines=None, n_noise_modes=8, manifold=None,
                 n_newton=12):
        self.n_components = n_components
        self.n_align = n_align
        self.max_shift = max_shift
        self.snr_min = snr_min
        self.covariance = covariance
        self.n_lines = n_lines
        self.n_noise_modes = n_noise_modes
        self.manifold = manifold or {}
        self.n_newton = n_newton
        self.calibration_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, traces, noise_traces=None, noise=None):
        """Learn all model components from unlabelled records.

        Provide either pulse-free ``noise_traces`` or a ready ``noise`` model.
        """
        traces = np.atleast_2d(np.asarray(traces, float))
        if noise is None:
            if noise_traces is None:
                raise ValueError("need noise_traces or noise")
            noise = StationaryNoise.from_records(noise_traces)
        self.noise_ = noise
        self.pca_ = ShiftInvariantPCA(self.n_components, self.n_align, self.max_shift,
                                      snr_min=self.snr_min).fit(traces, noise)
        self.manifold_ = PulseManifold(**self.manifold).fit(self.pca_.coeffs_)
        self.stationary_fitter_ = ManifoldFitter(noise, self.pca_.basis_, self.manifold_)
        s, t0, _ = self.stationary_fitter_.fit(traces, self.manifold_.s_, self.pca_.t0_,
                                               n_iter=self.n_newton)
        self.lowrank_ = None
        self.line_centers_ = groups = None
        if self.n_lines:
            groups, self.line_centers_ = find_clusters(s, self.n_lines)
        if self.covariance is not None:
            Xa = self._aligned(traces, t0)
            if self.covariance == "lines":
                if groups is None:
                    raise ValueError('covariance="lines" requires n_lines')
                self.lowrank_ = LowRankNoise(self.n_noise_modes).fit(
                    Xa, s, groups=groups, include=self.pca_.basis_)
            elif self.covariance == "curve":
                res = Xa - self.manifold_.curve(s) @ self.pca_.basis_
                self.lowrank_ = LowRankNoise(self.n_noise_modes).fit(
                    res, s, include=self.pca_.basis_)
            else:
                raise ValueError("covariance must be 'lines', 'curve' or None")
        self.fitter_ = ManifoldFitter(noise, self.pca_.basis_, self.manifold_, self.lowrank_)
        if self.lowrank_ is not None:
            s, t0, _ = self.fitter_.fit(traces, s, t0, n_iter=self.n_newton)
            if self.line_centers_ is not None:
                # line positions on the scale of the final estimator
                self.line_centers_ = np.array([np.median(s[groups == g])
                                               for g in range(len(self.line_centers_))])
        self.s_train_, self.t0_train_ = s, t0
        return self

    def _aligned(self, traces, t0):
        n = self.noise_.n_samples
        Yh = self.noise_.whiten_rfft(traces)
        return np.fft.irfft(shift_rfft(Yh, -np.asarray(t0, float), n), n=n, axis=1)

    # ------------------------------------------------------------ transform
    def transform(self, traces):
        """Estimate ``(s, t0)`` for each record by maximum likelihood."""
        traces = np.atleast_2d(np.asarray(traces, float))
        t0, a = self.pca_.transform(traces)
        s = self.manifold_.project(a)
        s, t0, nll = self.stationary_fitter_.fit(traces, s, t0, n_iter=self.n_newton)
        if self.lowrank_ is not None:
            s, t0, nll = self.fitter_.fit(traces, s, t0, n_iter=self.n_newton)
        res = PulseFit(s=s, t0=t0, nll=nll, coeffs=a)
        if self.calibration_ is not None:
            res.energy = self.energy(s)
        return res

    def fit_transform(self, traces, noise_traces=None, noise=None):
        return self.fit(traces, noise_traces, noise).transform(traces)

    # ---------------------------------------------------------- calibration
    def calibrate(self, s_lines, energies):
        """Set the monotone ``s -> E`` map from line positions and known energies."""
        self.calibration_ = EnergyCalibration(s_lines, energies)
        return self

    def energy(self, s):
        if self.calibration_ is None:
            raise RuntimeError("call calibrate() first")
        return self.calibration_(s)

    # ------------------------------------------------------------- products
    def template(self, s):
        """Noise-free pulse template(s) at manifold coordinate(s) ``s`` (raw units)."""
        c = self.manifold_.curve(np.atleast_1d(s)) @ self.pca_.basis_
        return self.noise_.color(c)

    def crb(self, s, **kw):
        """Cramér-Rao ``(sigma_s, sigma_t0)`` of the learned model at ``s``."""
        return self.fitter_.crb(s, **kw)

    def resolution(self, s, **kw):
        """Expected FWHM energy resolution at ``s`` (requires a calibration)."""
        from .calibration import FWHM
        return FWHM * self.crb(s, **kw)[0] * self.calibration_.gain(s)
