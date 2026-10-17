"""Energy-scale calibration and line-resolution utilities."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

__all__ = ["EnergyCalibration", "find_lines", "find_clusters", "line_resolution", "robust_sigma", "FWHM"]

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def find_lines(x, n_lines, bins=None, smooth=2, x_range=None):
    """Locate the ``n_lines`` most prominent peaks of the distribution of ``x``.

    Peaks are found on a smoothed histogram and refined to the centroid of the
    Gaussian core (iteratively clipped mean within half the distance to the
    neighbouring peaks).  Returns positions in increasing order; no labels or
    energies are needed.
    """
    x = np.asarray(x, float)
    rng = x_range or (np.quantile(x, 0.001), np.quantile(x, 0.999))
    bins = bins or int(np.clip(4 * np.sqrt(x.size), 100, 20000))
    h, e = np.histogram(x, bins=bins, range=rng)
    if smooth:
        k = np.exp(-0.5 * (np.arange(-4 * smooth, 4 * smooth + 1) / smooth) ** 2)
        h = np.convolve(h, k / k.sum(), mode="same")
    pk, prop = find_peaks(h, prominence=0)
    best = pk[np.argsort(prop["prominences"])[::-1][:n_lines]]
    c = 0.5 * (e[1:] + e[:-1])
    pos = np.sort(c[best])
    edges = np.r_[rng[0], 0.5 * (pos[1:] + pos[:-1]), rng[1]]
    out = []
    for i, p in enumerate(pos):
        half = min(p - edges[i], edges[i + 1] - p)
        out.append(robust_sigma(x[np.abs(x - p) < half])[0])
    return np.asarray(out)


def robust_sigma(x, clip=4.0, n_iter=10):
    """Iteratively clipped standard deviation (Gaussian core of a line)."""
    x = np.asarray(x, float)
    mu, sd = np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))
    for _ in range(n_iter):
        sel = np.abs(x - mu) < clip * sd
        mu, sd_new = x[sel].mean(), x[sel].std()
        if np.isclose(sd_new, sd):
            break
        sd = sd_new
    return mu, sd


def line_resolution(e_est, clip=4.0):
    """Centroid, FWHM and FWHM standard error of a mono-energetic line."""
    mu, sd = robust_sigma(e_est, clip)
    n = np.sum(np.abs(e_est - mu) < clip * sd)
    return mu, FWHM * sd, FWHM * sd / np.sqrt(2 * max(n - 1, 1))


class EnergyCalibration:
    """Monotone map from an estimator scale ``x`` to energy through the origin.

    Uses shape-preserving cubic interpolation (PCHIP) through ``(0, 0)`` and
    the measured line positions, extrapolated linearly with the end slopes.
    """

    def __init__(self, x_lines, energies, through_origin=True):
        x = np.asarray(x_lines, float)
        e = np.asarray(energies, float)
        order = np.argsort(x)
        x, e = x[order], e[order]
        if through_origin:
            x, e = np.r_[0.0, x], np.r_[0.0, e]
        self._p = PchipInterpolator(x, e, extrapolate=False)
        self._d = self._p.derivative()
        self._x, self._e = x, e

    def __call__(self, x):
        x = np.asarray(x, float)
        out = self._p(np.clip(x, self._x[0], self._x[-1]))
        lo, hi = x < self._x[0], x > self._x[-1]
        out = np.where(lo, self._e[0] + (x - self._x[0]) * self._d(self._x[0]), out)
        out = np.where(hi, self._e[-1] + (x - self._x[-1]) * self._d(self._x[-1]), out)
        return out

    def gain(self, x):
        """Local ``dE/dx``."""
        x = np.clip(np.asarray(x, float), self._x[0], self._x[-1])
        return self._d(x)


def find_clusters(x, n_lines, width=5.0, search=None, **kw):
    """Label records belonging to the ``n_lines`` most prominent mono-energetic peaks.

    Peaks are found with :func:`find_lines`; a record belongs to a peak if it
    lies within ``width`` robust standard deviations of it.  Returns
    ``(labels, centers)`` with ``labels = -1`` for unassigned records.  Only the
    existence of the lines is used -- their energies need not be known.
    """
    x = np.asarray(x, float)
    centers = find_lines(x, n_lines, **kw)
    gaps = np.diff(np.r_[x.min(), centers, x.max()])
    labels = np.full(x.size, -1)
    for i, c in enumerate(centers):
        half = search if search is not None else 0.25 * min(gaps[i], gaps[i + 1])
        near = np.abs(x - c) < half
        _, sd = robust_sigma(x[near] - c)
        labels[(np.abs(x - c) < width * sd) & (labels < 0)] = i
    return labels, centers
