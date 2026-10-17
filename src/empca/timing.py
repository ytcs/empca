"""Label-free correction of the sampling-phase (arrival-time) bias.

A pulse that is not band-limited is displaced, relative to a Fourier-shifted
template, by an amount that depends on where it falls between two samples.
The estimated arrival time is then ``tau_hat = t0 + w(E) + b(phi) + noise``
with a periodic bias ``b`` of the true sub-sample phase ``phi``.  Because true
arrival phases are uniformly distributed, ``b`` is identifiable from the
non-uniformity of the histogram of ``tau_hat mod 1`` alone: the map that makes
that histogram uniform inverts ``phi -> phi + b(phi)``.

The density is estimated with ``n_harmonics`` Fourier terms within groups of
records of (nearly) equal energy -- mono-energetic clusters, or narrow bins of
the manifold coordinate -- because the walk ``w(E)`` must be constant within a
group.  The statistical requirement is severe: a bias of amplitude ``b`` makes
the density non-uniform by about ``2 pi b``, whose estimate has standard error
``sqrt(2/n)``; reliable correction needs ``n >> 2 / (2 pi b)^2`` records per
group (about 10^5 for b = 0.004 samples).
"""
import numpy as np

__all__ = ["PhaseEqualizer"]


def _cdf_shift(phi, c, k):
    """CDF(phi) - phi for the density 1 + 2 Re sum_k c_k exp(2 pi i k phi)."""
    e = np.exp(2j * np.pi * np.outer(phi, k))
    return np.real(np.sum(c * (e - 1) / (1j * np.pi * k), axis=1))


class PhaseEqualizer:
    """Equalise the sub-sample phase histogram of arrival-time estimates.

    Parameters
    ----------
    n_harmonics : int
        Number of Fourier harmonics of the phase density.
    """

    def __init__(self, n_harmonics=3):
        self.n_harmonics = n_harmonics

    def fit(self, tau, groups):
        """Learn one correction per group (labels >= 0); returns self."""
        tau, groups = np.asarray(tau, float), np.asarray(groups)
        k = np.arange(1, self.n_harmonics + 1)
        self.k_ = k
        self.coef_, self.offset_, self.n_ = {}, {}, {}
        for g in np.unique(groups[groups >= 0]):
            phi = np.mod(tau[groups == g], 1.0)
            c = np.mean(np.exp(-2j * np.pi * np.outer(phi, k)), axis=0)
            d = _cdf_shift(phi, c, k)
            # remove the arbitrary rotation: zero mean displacement on the circle
            self.offset_[g] = np.angle(np.mean(np.exp(2j * np.pi * d))) / (2 * np.pi)
            self.coef_[g] = c
            self.n_[g] = phi.size
        return self

    def transform(self, tau, groups):
        """Corrected arrival times (records in unknown groups are unchanged)."""
        tau, groups = np.asarray(tau, float).copy(), np.asarray(groups)
        for g, c in self.coef_.items():
            m = groups == g
            phi = np.mod(tau[m], 1.0)
            tau[m] = tau[m] + _cdf_shift(phi, c, self.k_) - self.offset_[g]
        return tau

    def required_events(self, bias_amplitude, snr=5.0):
        """Events per group needed to detect a phase bias of the given amplitude."""
        return snr ** 2 * 2.0 / (2 * np.pi * bias_amplitude) ** 2
