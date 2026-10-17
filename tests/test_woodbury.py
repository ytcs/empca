import time

import numpy as np

from empca.covariance import LowRankNoise, woodbury_inverse, woodbury_logdet
from empca.fitting import ManifoldFitter
from empca.manifold import PulseManifold
from empca.shifts import shift


def test_woodbury_identities():
    rng = np.random.default_rng(0)
    N, K = 64, 5
    V = np.linalg.qr(rng.standard_normal((N, K)))[0].T
    A = rng.standard_normal((K, K))
    S = A @ A.T - 0.3 * np.eye(K)
    R = np.eye(N) + V.T @ S @ V
    assert np.allclose(woodbury_inverse(V, S), np.linalg.inv(R), atol=1e-10)
    assert np.isclose(woodbury_logdet(S), np.linalg.slogdet(R)[1])


class _FixedSigma(LowRankNoise):
    def __init__(self, V, S0):
        super().__init__(n_modes=V.shape[0])
        self.V_ = V
        self._S0 = S0
        self.floor = 0.9

    def sigma_eig(self, s):
        s = np.atleast_1d(s)
        S = np.einsum("n,kl->nkl", s / 10.0, self._S0)
        w, U = np.linalg.eigh(S)
        return np.clip(w, -self.floor, None), U


def test_fast_likelihood_equals_dense_shifted_covariance(noise_model):
    """-2 log L from the K_b-dimensional formula equals the dense N x N evaluation
    with R(s, tau) = I + T_tau V^T Sigma(s) V T_tau^T in the whitened domain."""
    rng = np.random.default_rng(1)
    N = noise_model.n_samples
    band = np.fft.irfft(np.fft.rfft(np.eye(N), axis=1) * noise_model.mask, n=N, axis=1)
    Bs = np.linalg.qr(band @ rng.standard_normal((N, 3)))[0].T
    a = np.stack([np.linspace(0, 1, 50) * 30, 5 * np.linspace(0, 1, 50) ** 2, np.zeros(50)], 1) @ \
        np.linalg.qr(rng.standard_normal((3, 3)))[0]
    man = PulseManifold(n_knots=6).fit(a + 0.01 * rng.standard_normal(a.shape))
    V = np.linalg.qr(band @ rng.standard_normal((N, 4)))[0].T
    S0 = np.diag([3.0, 1.0, 0.5, -0.4])
    lr = _FixedSigma(V, S0)
    fit = ManifoldFitter(noise_model, Bs, man, lr)

    s, tau = 12.0, 1.37
    y = rng.standard_normal(N)
    raw = noise_model.color(y)
    fast = fit.nll(raw[None], [s], [tau])[0]

    yw = noise_model.whiten(raw)
    mu = shift(man.curve([s])[0] @ Bs, tau)
    TV = shift(V, tau)
    S = lr.sigma([s])[0]
    # restrict to the analysis band (DC and Nyquist excluded)
    Q = np.linalg.qr(band)[0][:, : noise_model.dof]
    R = Q.T @ (np.eye(N) + TV.T @ S @ TV) @ Q
    r = Q.T @ (yw - mu)
    L = np.linalg.cholesky(R)
    z = np.linalg.solve(L, r)
    dense = z @ z + 2 * np.sum(np.log(np.diag(L)))
    assert np.isclose(fast, dense, rtol=1e-9, atol=1e-8)
