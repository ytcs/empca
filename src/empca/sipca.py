"""Shift-invariant whitened PCA (Stage A).

Model, in the whitened domain (unit white noise),

    y_j = T_{tau_j} B^T a_j + n_j,      B: (K, N) orthonormal rows,

with unknown arrival times ``tau_j`` and amplitudes ``a_j``.  The algorithm is
a hard (classification) EM / block coordinate ascent on the captured power
``sum_j ||B T_{-tau_j} y_j||^2``:

* E-step: for fixed ``B`` each ``tau_j`` maximises ``||B T_{-tau} y_j||^2``.
  All integer lags are scored with one phase-coherent inverse FFT per basis
  vector; the maximum is refined to sub-sample precision with Newton steps on
  the exact band-limited correlation (analytic derivatives in frequency).
* M-step: for fixed ``tau`` the records are counter-rotated (``T_{-tau_j}``,
  a phase ramp in frequency) and ``B`` is the top-K right singular subspace.

Both steps can only increase the objective, so it converges monotonically.
Because the noise is white and isotropic after whitening, the noise adds a
multiple of the identity to the scatter matrix and does not bias the learned
subspace (only the alignment does, at low SNR; see ``snr_min``).
"""
import numpy as np

from .noise import rfft_weights
from .shifts import xcorr, aligned_projections, shift_rfft

__all__ = ["ShiftInvariantPCA", "refine_peak"]


def refine_peak(Yh, Bh, tau, n, n_iter=6, bound=None):
    """Sub-sample maximisation of ``g(tau) = ||x(tau)||^2`` by guarded Newton steps."""
    tau = np.asarray(tau, float).copy()
    g_best = None
    for _ in range(n_iter):
        x, dx, ddx = aligned_projections(Yh, Bh, tau, n, derivs=2)
        g = np.sum(x * x, axis=1)
        g1 = 2 * np.sum(x * dx, axis=1)
        g2 = 2 * np.sum(dx * dx + x * ddx, axis=1)
        if g_best is None:
            g_best, tau_best = g.copy(), tau.copy()
        else:
            better = g > g_best
            g_best = np.where(better, g, g_best)
            tau_best = np.where(better, tau, tau_best)
        step = np.where(g2 < 0, -g1 / np.where(g2 < 0, g2, -1.0), 0.25 * np.sign(g1))
        step = np.clip(step, -0.5, 0.5)
        tau = tau + step
        if bound is not None:
            tau = np.clip(tau, -bound, bound)
    x = aligned_projections(Yh, Bh, tau, n)[0]
    g = np.sum(x * x, axis=1)
    better = g > g_best
    return np.where(better, tau, tau_best), np.where(better, g, g_best)


class ShiftInvariantPCA:
    """Shift-invariant PCA of pre-whitened pulse records.

    Parameters
    ----------
    n_components : int
        Dimension ``K`` of the pulse subspace.
    n_align : int
        Number ``r <= K`` of leading components used for alignment in the
        E-step.  The remaining components are learned after alignment and are
        not allowed to absorb time shifts (a subspace containing the time
        derivative of the pulse makes the arrival time unidentifiable).  The
        objective ``J_r = sum_j ||B_r T_{-tau_j} y_j||^2`` is non-decreasing
        because the leading ``r`` singular vectors maximise it in the M-step.
    max_shift : int
        Arrival-time search window, in samples, around the current gauge.
    n_iter : int
        Maximum number of E/M iterations.
    tol : float
        Relative change of the captured power used as stopping criterion.
    snr_min : float, optional
        Only records whose whitened excess power exceeds ``snr_min**2`` are
        used to learn the basis (all records are aligned and projected).  This
        suppresses the "template from noise" bias of correlation alignment.

    Attributes
    ----------
    basis_ : ndarray (K, N)
        Orthonormal whitened basis (rows), counter-rotated to a common gauge.
    t0_ : ndarray (n,)
        Arrival-time offsets (samples) of the training records in that gauge.
    coeffs_ : ndarray (n, K)
        Aligned projections ``B T_{-t0} y_w``.
    singular_values_, objective_ : diagnostics.
    """

    def __init__(self, n_components=4, n_align=1, max_shift=16, n_iter=30, tol=1e-9, snr_min=None):
        self.n_components = n_components
        self.n_align = n_align
        self.max_shift = max_shift
        self.n_iter = n_iter
        self.tol = tol
        self.snr_min = snr_min

    # ------------------------------------------------------------------ fit
    def fit(self, traces, noise):
        """Learn the whitened shift-invariant basis from raw records.

        Parameters
        ----------
        traces : ndarray (n, N)
        noise : :class:`empca.noise.StationaryNoise`
        """
        self.noise_ = noise
        n = noise.n_samples
        Yh = noise.whiten_rfft(np.asarray(traces, float))
        power = np.sum(np.abs(Yh) ** 2 * rfft_weights(n), axis=1) / n
        excess = power - noise.dof
        train = np.ones(len(Yh), bool) if self.snr_min is None else excess > self.snr_min ** 2
        if train.sum() < self.n_components:
            raise ValueError("too few records above snr_min to learn the basis")
        self.train_mask_ = train

        # initialise with the unaligned SVD (sufficient for jitter << pulse width)
        tau = np.zeros(len(Yh))
        B, _ = self._m_step(Yh[train], tau[train], n)
        self.objective_ = []
        for _ in range(self.n_iter):
            tau_prev = tau
            tau, gain = self._e_step(Yh, B, n)
            B, sv = self._m_step(Yh[train], tau[train], n)
            # gauge: median arrival time of the training set is zero
            med = np.median(tau[train])
            B = np.fft.irfft(shift_rfft(np.fft.rfft(B, axis=1), med, n), n=n, axis=1)
            tau = tau - med
            obj = float(np.sum(sv[: self.n_align] ** 2))
            self.objective_.append(obj)
            moved = np.max(np.abs(tau - tau_prev + np.median(tau_prev - tau)))
            if len(self.objective_) > 1 and (abs(obj - self.objective_[-2]) <= self.tol * obj
                                             or moved < 1e-4):
                break
        tau, _ = self._e_step(Yh, B, n)
        B = self._fix_signs(B, Yh[train], tau[train], n)
        self.basis_ = B
        self.singular_values_ = sv
        self.t0_ = tau
        self.coeffs_ = aligned_projections(Yh, np.fft.rfft(B, axis=1), tau, n)[0]
        return self

    def _e_step(self, Yh, B, n):
        B = B[: self.n_align]
        lags, P = xcorr(Yh, B, self.max_shift)
        score = np.sum(P * P, axis=1)
        tau0 = lags[np.argmax(score, axis=1)].astype(float)
        return refine_peak(Yh, np.fft.rfft(B, axis=1), tau0, n, bound=self.max_shift + 1)

    def _m_step(self, Yh, tau, n):
        X = np.fft.irfft(shift_rfft(Yh, -tau, n), n=n, axis=1)
        # top-K right singular vectors via the (N x N) scatter matrix
        w, U = np.linalg.eigh(X.T @ X)
        idx = np.argsort(w)[::-1][: self.n_components]
        return U[:, idx].T, np.sqrt(np.clip(w[idx], 0, None))

    @staticmethod
    def _fix_signs(B, Yh, tau, n):
        a = aligned_projections(Yh, np.fft.rfft(B, axis=1), tau, n)[0]
        s = np.sign(np.sum(a * np.abs(a[:, :1]), axis=0))
        s[s == 0] = 1
        return B * s[:, None]

    # ------------------------------------------------------------ transform
    def transform(self, traces):
        """Align and project new records: returns ``(t0, coeffs)``."""
        n = self.noise_.n_samples
        Yh = self.noise_.whiten_rfft(np.asarray(traces, float))
        tau, _ = self._e_step(Yh, self.basis_, n)
        a = aligned_projections(Yh, np.fft.rfft(self.basis_, axis=1), tau, n)[0]
        return tau, a

    def templates(self):
        """Basis vectors mapped back to the raw (coloured) domain."""
        return self.noise_.color(self.basis_)

