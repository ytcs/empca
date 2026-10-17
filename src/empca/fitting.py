"""Joint maximum-likelihood energy/time fit on the pulse manifold (Stage C).

Whitened model for one record ``y_w`` with parameters ``(s, tau)``:

    y_w ~ N( T_tau B_s^T c(s),  I + T_tau V^T Sigma(s) V T_tau^T ).

Because whitening and shifting commute and ``T_tau`` is orthogonal, the
likelihood can be evaluated in the frame aligned to ``tau`` where the
covariance no longer depends on ``tau``.  Let ``B`` be an orthonormal basis of
``span(B_s, V)`` (dimension ``K_b = K_s + K_n`` at most), ``x(tau) = B T_{-tau} y_w``
its aligned projections, ``m(s) = B B_s^T c(s)`` and ``Q = B V^T``.  The
Woodbury identity and the determinant lemma reduce ``-2 log L`` to

    ||y_w||^2 - ||x||^2 + r^T r - (Q^T r)^T M(s) (Q^T r) + log det(I + Sigma(s)),
    r = x - m(s),   M = Sigma (I + Sigma)^{-1},

an expression in ``K_b`` dimensions.  ``x(tau)`` and its tau-derivatives cost
``O(K_b N)`` per record for continuous ``tau`` (or ``O(K_b N log N)`` for all
integer lags at once), so the full non-stationary likelihood costs no more
than a stationary optimal filter with ``K_b`` templates.
"""
import numpy as np

from .noise import rfft_weights
from .shifts import aligned_projections, derivative

__all__ = ["ManifoldFitter"]


class ManifoldFitter:
    """Maximum-likelihood (s, t0) estimation on a learned pulse manifold.

    Parameters
    ----------
    noise : StationaryNoise
    basis : ndarray (K_s, N), whitened orthonormal basis (rows) of the manifold
    manifold : PulseManifold
    lowrank : LowRankNoise, optional
        Non-stationary excess covariance; ``None`` gives the stationary ML fit.
    """

    def __init__(self, noise, basis, manifold, lowrank=None):
        self.noise = noise
        self.manifold = manifold
        self.lowrank = lowrank
        Bs = np.asarray(basis, float)
        self.n_s = Bs.shape[0]
        if lowrank is None:
            B = Bs
            self.Q = None
        else:
            V = lowrank.V_
            Vp = V - (V @ Bs.T) @ Bs
            U, sv, Wt = np.linalg.svd(Vp, full_matrices=False)
            Bp = Wt[sv > 1e-8 * max(sv.max(), 1.0)]
            B = np.vstack([Bs, Bp])
            self.Q = B @ V.T  # (K_b, K_n)
        self.B = B
        self.Bh = np.fft.rfft(B, axis=1)
        self.n = noise.n_samples

    # ------------------------------------------------------------- model
    def _model(self, s):
        c = self.manifold.curve(s)
        m = np.zeros((len(s), self.B.shape[0]))
        m[:, : self.n_s] = c
        if self.Q is None:
            return m, None, None, np.zeros(len(s))
        w, U = self.lowrank.sigma_eig(s)
        return m, w / (1.0 + w), U, np.sum(np.log1p(w), axis=1)

    def _f(self, x, s, grad=False):
        """``-2 log L - ||y_w||^2`` and its gradient in ``x``."""
        m, mw, U, ld = self._model(s)
        r = x - m
        f = -np.sum(x * x, axis=1) + np.sum(r * r, axis=1) + ld
        if self.Q is not None:
            z = np.einsum("nkl,nk->nl", U, r @ self.Q)  # U^T Q^T r
            f -= np.sum(mw * z * z, axis=1)
        if not grad:
            return f
        g = -2 * x + 2 * r
        if self.Q is not None:
            g -= 2 * (np.einsum("nkl,nl->nk", U, mw * z) @ self.Q.T)
        return f, g

    def _curv_tt(self, s, dx):
        if self.Q is None:
            return 0.0
        _, mw, U, _ = self._model(s)
        z = np.einsum("nkl,nk->nl", U, dx @ self.Q)
        return -2 * np.sum(mw * z * z, axis=1)

    # ---------------------------------------------------------- interface
    def nll(self, traces, s, tau):
        """``-2 log L`` up to a constant (``||y_w||^2`` included)."""
        Yh = self.noise.whiten_rfft(np.atleast_2d(traces))
        yy = self._power(Yh)
        x = aligned_projections(Yh, self.Bh, np.asarray(tau, float), self.n)[0]
        return yy + self._f(x, np.asarray(s, float))

    def _power(self, Yh):
        return np.sum(np.abs(Yh) ** 2 * rfft_weights(self.n), axis=1) / self.n

    def fit(self, traces, s0, tau0, n_iter=12, h=0.05, max_ds=10.0, batch=8192):
        """Damped Newton maximisation of the likelihood from ``(s0, tau0)``.

        Returns ``s, tau, nll`` (``nll`` = ``-2 log L`` up to a constant).
        """
        traces = np.atleast_2d(traces)
        out = [self._fit_batch(traces[b:b + batch], np.asarray(s0, float)[b:b + batch],
                               np.asarray(tau0, float)[b:b + batch], n_iter, h, max_ds)
               for b in range(0, len(traces), batch)]
        return tuple(np.concatenate(o) for o in zip(*out))

    def _fit_batch(self, traces, s, tau, n_iter, h, max_ds):
        Yh = self.noise.whiten_rfft(traces)
        s, tau = s.copy(), tau.copy()
        n = self.n
        for _ in range(n_iter):
            x, dx, ddx = aligned_projections(Yh, self.Bh, tau, n, derivs=2)
            f0, g0 = self._f(x, s, grad=True)
            fp, gp = self._f(x, s + h, grad=True)
            fm, gm = self._f(x, s - h, grad=True)
            fs = (fp - fm) / (2 * h)
            fss = (fp - 2 * f0 + fm) / h ** 2
            ft = np.sum(g0 * dx, axis=1)
            ftt = self._curv_tt(s, dx) + np.sum(g0 * ddx, axis=1)
            fst = np.sum((gp - gm) * dx, axis=1) / (2 * h)
            # Levenberg-damped 2x2 Newton step
            tr, det = fss + ftt, fss * ftt - fst ** 2
            lam_min = 0.5 * tr - np.sqrt(np.maximum(0.25 * tr ** 2 - det, 0))
            mu = np.where(lam_min > 0, 0.0, -lam_min + 1e-3 * (np.abs(tr) + 1))
            a, d = fss + mu, ftt + mu
            det = a * d - fst ** 2
            ds = -(d * fs - fst * ft) / det
            dt = -(a * ft - fst * fs) / det
            ds = np.clip(ds, -max_ds, max_ds)
            dt = np.clip(dt, -0.5, 0.5)
            done = np.zeros(len(s), bool)
            for _ in range(4):
                st, tt = s + ds, tau + dt
                xt = aligned_projections(Yh, self.Bh, tt, n)[0]
                ok = (self._f(xt, st) < f0) & ~done
                s[ok], tau[ok] = st[ok], tt[ok]
                done |= ok
                ds, dt = 0.5 * ds, 0.5 * dt
            if done.sum() == 0:
                break
        x = aligned_projections(Yh, self.Bh, tau, n)[0]
        return s, tau, self._power(Yh) + self._f(x, s)

    # ------------------------------------------------------------- Fisher
    def fisher(self, s, include_covariance=True, h=0.05):
        """Fisher information matrix for ``(s, tau)`` at manifold point ``s``.

        Includes the mean term and (optionally) the covariance term
        ``1/2 tr(R^-1 R_i R^-1 R_j)`` of the learned non-stationary model.
        """
        Bs = self.B[: self.n_s]
        c = self.manifold.curve([s])[0]
        mu = c @ Bs
        d_s = self.manifold.curve([s], 1)[0] @ Bs
        d_t = -derivative(mu)
        N = self.n
        if self.lowrank is None:
            Rinv = np.eye(N)
            Rs = Rt = None
        else:
            V = self.lowrank.V_
            Sg = self.lowrank.sigma([s])[0]
            Rinv = np.eye(N) - V.T @ (Sg @ np.linalg.inv(np.eye(len(Sg)) + Sg)) @ V
            dSg = (self.lowrank.sigma([s + h])[0] - self.lowrank.sigma([s - h])[0]) / (2 * h)
            Rs = V.T @ dSg @ V
            DV = derivative(V)
            Rt = -(DV.T @ Sg @ V + V.T @ Sg @ DV)
        D = np.vstack([d_s, d_t])
        F = D @ Rinv @ D.T
        if include_covariance and Rs is not None:
            A = [Rinv @ Rs, Rinv @ Rt]
            for i in range(2):
                for j in range(2):
                    F[i, j] += 0.5 * np.sum(A[i] * A[j].T)
        return F

    def crb(self, s, **kw):
        """Cramér-Rao standard deviations ``(sigma_s, sigma_tau)`` at ``s``."""
        F = self.fisher(s, **kw)
        return np.sqrt(np.diag(np.linalg.inv(F)))
