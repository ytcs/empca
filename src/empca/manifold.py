"""Spline pulse manifold in the whitened PCA coefficient space (Stage B).

The aligned, whitened coefficients ``a_j`` (K-dimensional, unit isotropic
noise) of a family of pulses lie close to a smooth curve that starts at the
origin (zero energy) and bends as the detector saturates.

Representation
--------------
Let ``u = <e, a>`` be the coordinate along the leading (uncentred) principal
direction ``e`` of the coefficients -- the mean-pulse direction, along which
the projection grows monotonically with energy -- and ``w`` the remaining
``K - 1`` rotated coordinates.  The curve is the graph

    c(u) = e u + W^T u h(u),     h: cubic P-spline (Eilers & Marx 1996),

which passes through the origin, is linear in the small-signal limit and,
being a graph over ``u``, cannot fold back on itself (a failure mode of
free-form principal curves in sparse, strongly saturated regions).  For
``u < 0`` the curve is continued linearly through the origin, so the
projection of pure noise is *unbiased* (no rectification as produced by the
norm of a coefficient vector); beyond the data it is continued along its
tangent.

Parameterisation
----------------
The curve is re-parameterised by whitened arc length ``s`` (``||dc/ds|| = 1``),
so ``s`` is the cumulative matched-filter SNR along the manifold.  Under
stationary noise the Fisher information of ``s`` is exactly 1, hence
``sigma_E = dE/ds``.  The coordinate of a record is the closest point of the
curve (the stationary-noise maximum-likelihood estimate given the alignment).
"""
import numpy as np
from scipy.interpolate import BSpline

__all__ = ["PulseManifold"]


def _weighted_quantile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return x[order][np.searchsorted(cw, q * cw[-1])]


class PulseManifold:
    """One-dimensional pulse manifold (smooth curve through the origin).

    Parameters
    ----------
    n_knots : int
        Number of interior spline knots (uniform in ``u``).
    penalty : float or "gcv"
        Second-difference P-spline penalty relative to
        ``tr(Phi^T Phi) / tr(D^T D)``; ``"gcv"`` selects it by generalised
        cross-validation.
    end_quantile : float
        The spline is supported up to this quantile of ``u``; beyond it the
        curve is continued along its tangent.
    grid_size : int
        Size of the coarse grid used to initialise each projection.
    min_free_per_knot : int
        When clusters are given and fewer than ``min_free_per_knot`` records per
        spline coefficient lie *outside* the clusters, the data cannot determine
        the curvature between lines.  The curve is then the secant through the
        origin and the cluster means (``mode_ = "secant"``), the learned analogue
        of the tangent filter.  Otherwise ``mode_ = "spline"``.
    """

    def __init__(self, n_knots=16, penalty="gcv", end_quantile=0.995, grid_size=4096,
                 min_free_per_knot=30):
        self.min_free_per_knot = min_free_per_knot
        self.n_knots = n_knots
        self.penalty = penalty
        self.end_quantile = end_quantile
        self.grid_size = grid_size

    # ---------------------------------------------------------------- fit
    def fit(self, A, groups=None):
        """Fit the manifold to whitened, aligned coefficients ``A`` (n, K).

        ``groups`` (optional, one int per row, ``-1`` = none) marks records that
        belong to mono-energetic clusters.  Such clusters enter the curve fit only
        through their mean (with weight equal to their size): their internal
        scatter is, by definition, not along the energy direction and must not
        bend the curve (see the identifiability discussion in the paper).
        """
        A_all = np.atleast_2d(np.asarray(A, float))
        A, wts = A_all, np.ones(len(A_all))
        if groups is not None:
            groups = np.asarray(groups)
            free = groups < 0
            labels = np.unique(groups[~free])
            A = np.vstack([A_all[free]] + [A_all[groups == g].mean(0, keepdims=True) for g in labels])
            wts = np.r_[np.ones(free.sum()), [np.sum(groups == g) for g in labels]].astype(float)
        Rt = np.linalg.svd(A * np.sqrt(wts)[:, None], full_matrices=False)[2]
        if np.median(A @ Rt[0]) < 0:
            Rt[0] = -Rt[0]
        self.rotation_ = Rt  # rows: e, then the K-1 orthogonal directions
        Z = A @ Rt.T
        u, w = Z[:, 0], Z[:, 1:]
        self.u_max_ = _weighted_quantile(u, wts, self.end_quantile)
        n_free = np.sum((wts == 1) & (u > 0) & (u <= self.u_max_))
        if groups is not None and n_free < self.min_free_per_knot * (self.n_knots + 4):
            self._fit_secant(u[wts > 1], w[wts > 1])
            self._build_arclength()
            self.s_, self.distance_ = self.project(A_all, return_distance=True)
            return self
        self.mode_ = "spline"
        k = 3
        inner = np.linspace(0.0, self.u_max_, self.n_knots + 2)
        self._t = np.r_[[0.0] * k, inner, [self.u_max_] * k]
        nb = len(self._t) - k - 1
        keep = (u > 0) & (u <= self.u_max_)
        uk, sw = u[keep], np.sqrt(wts[keep])[:, None]
        Phi = BSpline.design_matrix(uk, self._t, k).toarray() * uk[:, None] * sw
        Wk = w[keep] * sw
        D2 = np.diff(np.eye(nb), 2, axis=0)
        PtP, PtW, P2 = Phi.T @ Phi, Phi.T @ Wk, D2.T @ D2
        base = np.trace(PtP) / np.trace(P2)
        if self.penalty == "gcv":
            ww = np.sum(Wk ** 2)
            best = None
            for lam in base * np.logspace(-9, 2, 23):
                Minv = np.linalg.inv(PtP + lam * P2)
                coef = Minv @ PtW
                rss = ww - 2 * np.sum(coef * PtW) + np.sum(coef * (PtP @ coef))
                gcv = rss / (keep.sum() - np.trace(Minv @ PtP)) ** 2
                if best is None or gcv < best[0]:
                    best = (gcv, lam, coef)
            self.lambda_, coef = best[1], best[2]
        else:
            self.lambda_ = self.penalty * base
            coef = np.linalg.solve(PtP + self.lambda_ * P2, PtW)
        self._h = [BSpline(self._t, coef, k, extrapolate=False)]
        self._h += [self._h[0].derivative(1), self._h[0].derivative(2)]
        self._build_arclength()
        self.s_, self.distance_ = self.project(A_all, return_distance=True)
        return self

    def _fit_secant(self, ug, wg):
        """Curve through the origin and the cluster means: c = [u, u h(u)] with h
        piecewise linear between the cluster slopes w_g / u_g (constant outside)."""
        self.mode_ = "secant"
        self.lambda_ = np.nan
        order = np.argsort(ug)
        ug, hg = ug[order], wg[order] / ug[order, None]
        self.u_max_ = ug[-1]
        ux = np.r_[0.0, ug]
        hv = np.vstack([hg[:1], hg])
        slopes = np.diff(hv, axis=0) / np.maximum(np.diff(ux), 1e-12)[:, None]

        def h0(x):
            return np.stack([np.interp(x, ux, hv[:, j]) for j in range(hv.shape[1])], 1)

        def h1(x):
            i = np.clip(np.searchsorted(ux, x, side="right") - 1, 0, len(slopes) - 1)
            return slopes[i]

        def h2(x):
            return np.zeros((np.size(x), hv.shape[1]))

        self._h = [h0, h1, h2]

    # ------------------------------------------------------ curve in u
    def _curve_u(self, u, deriv=0):
        """Rotated-frame curve ``[u, u h(u)]`` and u-derivatives, continued linearly."""
        u = np.atleast_1d(np.asarray(u, float))
        uc = np.clip(u, 0.0, self.u_max_)
        h, h1, h2 = (spl(uc) for spl in self._h)
        out = np.zeros((u.size, h.shape[1] + 1))
        if deriv == 0:
            out[:, 0] = uc
            out[:, 1:] = uc[:, None] * h
            lo, hi = u < 0, u > self.u_max_
            if lo.any():
                out[lo] = u[lo, None] * self._curve_u(np.zeros(1), 1)[0]
            if hi.any():
                dM = self._curve_u(np.array([self.u_max_]), 1)[0]
                out[hi] += (u[hi] - self.u_max_)[:, None] * dM
        elif deriv == 1:
            out[:, 0] = 1.0  # clipped evaluation gives the end tangents outside
            out[:, 1:] = h + uc[:, None] * h1
        else:
            out[:, 1:] = 2 * h1 + uc[:, None] * h2
            out[(u < 0) | (u > self.u_max_)] = 0.0
        return out

    # ------------------------------------------------------ arc length
    def _build_arclength(self, n_grid=16384):
        ug = np.linspace(0.0, self.u_max_, n_grid)
        speed = np.linalg.norm(self._curve_u(ug, 1), axis=1)
        S = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(ug))])
        self._ug, self._sg, self._speed = ug, S, speed
        self.s_max_ = S[-1]

    def u_of_s(self, s):
        s = np.asarray(s, float)
        out = np.interp(s, self._sg, self._ug)
        out = np.where(s < 0, s / self._speed[0], out)
        return np.where(s > self.s_max_, self.u_max_ + (s - self.s_max_) / self._speed[-1], out)

    def s_of_u(self, u):
        u = np.asarray(u, float)
        out = np.interp(u, self._ug, self._sg)
        out = np.where(u < 0, u * self._speed[0], out)
        return np.where(u > self.u_max_, self.s_max_ + (u - self.u_max_) * self._speed[-1], out)

    # ------------------------------------------------------ public curve
    def curve(self, s, deriv=0):
        """``c(s)`` (deriv=0), ``c'(s)`` (1) or ``c''(s)`` (2) in coefficient space."""
        s = np.atleast_1d(np.asarray(s, float))
        u = self.u_of_s(s)
        if deriv == 0:
            out = self._curve_u(u)
        else:
            cu = self._curve_u(u, 1)
            nrm = np.linalg.norm(cu, axis=1)
            if deriv == 1:
                out = cu / nrm[:, None]
            else:
                cuu = self._curve_u(u, 2)
                uss = -np.sum(cu * cuu, axis=1) / nrm ** 4
                out = cuu / nrm[:, None] ** 2 + cu * uss[:, None]
        return out @ self.rotation_

    def speed(self, s):
        """``||dc/ds||`` (identically 1 by construction, up to discretisation)."""
        return np.linalg.norm(self.curve(s, 1), axis=1)

    # ------------------------------------------------------------ project
    def project(self, A, n_newton=5, return_distance=False):
        """Closest-point (stationary-noise ML) arc-length coordinate of rows ``A``."""
        Z = np.atleast_2d(A) @ self.rotation_.T
        lo, hi = -0.05 * self.u_max_, 1.05 * self.u_max_
        grid = np.linspace(lo, hi, self.grid_size)
        C = self._curve_u(grid)
        cc = np.sum(C * C, axis=1)
        u = np.empty(len(Z))
        for b in range(0, len(Z), 8192):
            u[b:b + 8192] = grid[np.argmin(cc[None, :] - 2 * Z[b:b + 8192] @ C.T, axis=1)]
        d0 = self._curve_u(np.zeros(1), 1)[0]
        lin = Z @ d0 / (d0 @ d0)
        u = np.where(lin < lo, lin, u)
        step = (hi - lo) / (self.grid_size - 1)
        for _ in range(n_newton):
            r = Z - self._curve_u(u)
            d1, d2 = self._curve_u(u, 1), self._curve_u(u, 2)
            f1 = -2 * np.sum(r * d1, axis=1)
            f2 = 2 * (np.sum(d1 * d1, axis=1) - np.sum(r * d2, axis=1))
            du = np.where(f2 > 0, -f1 / np.where(f2 > 0, f2, 1.0), 0.0)
            u = u + np.clip(du, -step, step)
        s = self.s_of_u(u)
        if return_distance:
            return s, np.linalg.norm(Z - self._curve_u(u), axis=1)
        return s
