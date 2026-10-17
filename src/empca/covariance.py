"""Low-rank non-stationary noise model and Woodbury identities (Stage C).

In the whitened frame aligned to the pulse, the residual covariance about the
manifold template is modelled as

    R_w(s) = I + V^T Sigma(s) V,      V: (K_n, N) orthonormal rows,

i.e. in the raw domain ``R(E, t0) = R0 + T_t0 R0^{1/2} V^T Sigma V R0^{1/2} T_t0^T``.
``V`` holds the dominant pulse-correlated noise modes (pooled over energies)
and ``Sigma(s)`` their energy-dependent ``K_n x K_n`` excess covariance,
interpolated smoothly in the manifold coordinate ``s`` with ``Sigma(0) = 0``.

The Woodbury identity and the matrix determinant lemma give

    R_w^{-1} = I - V^T [Sigma (I + Sigma)^{-1}] V
    log det R_w = log det(I + Sigma),

so no ``N x N`` matrix is ever formed or factorised.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["LowRankNoise", "woodbury_inverse", "woodbury_logdet"]


def woodbury_inverse(V, Sigma):
    """Dense ``(I + V^T Sigma V)^{-1}`` for orthonormal rows ``V`` (testing helper)."""
    K = Sigma.shape[0]
    M = Sigma @ np.linalg.inv(np.eye(K) + Sigma)
    return np.eye(V.shape[1]) - V.T @ M @ V


def woodbury_logdet(Sigma):
    """``log det(I + V^T Sigma V) = log det(I + Sigma)`` for orthonormal rows ``V``."""
    return np.linalg.slogdet(np.eye(Sigma.shape[-1]) + Sigma)[1]


class LowRankNoise:
    """Energy-dependent low-rank excess covariance in the whitened aligned frame.

    Parameters
    ----------
    n_modes : int
        Number of non-stationary noise modes ``K_n``.
    n_bins : int
        Number of quantile bins in ``s`` used to estimate ``Sigma(s)``.
    s_min : float
        Only residuals with ``s > s_min`` are used (low-energy records carry no
        information on the excess).
    floor : float
        Eigenvalues of ``Sigma`` are clipped to ``>= -floor`` (``R_w`` stays
        positive definite for ``floor < 1``).
    """

    def __init__(self, n_modes=8, n_bins=24, s_min=5.0, floor=0.5):
        self.n_modes = n_modes
        self.n_bins = n_bins
        self.s_min = s_min
        self.floor = floor

    def fit(self, residuals, s, groups=None):
        """Estimate ``V`` and ``Sigma(s)`` from aligned whitened residuals (n, N).

        Parameters
        ----------
        residuals : ndarray (n, N)
            Whitened residuals in the frame aligned to each pulse.  Residuals
            about the fitted curve (label-free) only reveal fluctuations
            orthogonal to the manifold tangent; residuals about the mean of a
            mono-energetic cluster (``groups``) also reveal the tangential
            component, which is the one that broadens the energy estimate.
        s : ndarray (n,)
            Manifold coordinate of each record.
        groups : ndarray of int (n,), optional
            Cluster label per record (``-1``: unused).  Each cluster gives one
            knot of ``Sigma(s)`` at its median ``s`` and residuals are taken
            about the cluster mean.  Without groups, quantile bins in ``s`` are
            used and the residuals are used as given.
        """
        residuals = np.asarray(residuals, float)
        s = np.asarray(s, float)
        if groups is not None:
            groups = np.asarray(groups)
            use = groups >= 0
            R, s_, g_ = residuals[use].copy(), s[use], groups[use]
            for g in np.unique(g_):
                R[g_ == g] -= R[g_ == g].mean(axis=0)
        else:
            use = s > self.s_min
            R, s_ = residuals[use], s[use]
            edges = np.quantile(s_, np.linspace(0, 1, self.n_bins + 1))
            g_ = np.clip(np.searchsorted(edges, s_, side="right") - 1, 0, self.n_bins - 1)
        C = R.T @ R / len(R)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1][: self.n_modes]
        self.V_ = evecs[:, order].T
        self.pooled_eigenvalues_ = evals[order]
        Z = R @ self.V_.T  # mode amplitudes (n, K_n)

        centers, sig = [0.0], [np.zeros((self.n_modes, self.n_modes))]
        for g in np.unique(g_):
            m = g_ == g
            if m.sum() < 4 * self.n_modes:
                continue
            centers.append(np.median(s_[m]))
            sig.append(Z[m].T @ Z[m] / m.sum() - np.eye(self.n_modes))
        order = np.argsort(centers)
        centers = [centers[i] for i in order]
        sig = [sig[i] for i in order]
        self.s_knots_ = np.asarray(centers)
        self.sigma_knots_ = np.asarray(sig)
        self._interp = PchipInterpolator(self.s_knots_, self.sigma_knots_, axis=0, extrapolate=False)
        return self

    def sigma_eig(self, s):
        """Eigen-decomposition ``(w, U)`` of ``Sigma(s)`` with the floor applied."""
        s = np.clip(np.atleast_1d(np.asarray(s, float)), self.s_knots_[0], self.s_knots_[-1])
        S = self._interp(s)
        S = 0.5 * (S + np.swapaxes(S, 1, 2))
        w, U = np.linalg.eigh(S)
        return np.clip(w, -self.floor, None), U

    def sigma(self, s):
        """Excess covariance ``Sigma(s)``, shape (len(s), K_n, K_n)."""
        w, U = self.sigma_eig(s)
        return (U * w[:, None, :]) @ np.swapaxes(U, 1, 2)

    def explained_excess(self, residuals, s):
        """Fraction of the total excess variance captured by the modes, per record."""
        R = np.asarray(residuals, float)
        tot = np.sum(R * R, axis=1)
        cap = np.sum((R @ self.V_.T) ** 2, axis=1)
        return cap, tot

    def dense_covariance(self, s):
        """Dense whitened ``I + V^T Sigma(s) V`` (testing / Fisher information)."""
        S = self.sigma(s)[0]
        return np.eye(self.V_.shape[1]) + self.V_.T @ S @ self.V_
