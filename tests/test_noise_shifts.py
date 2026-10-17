import numpy as np

from empca.noise import StationaryNoise
from empca.shifts import aligned_projections, derivative, shift, xcorr


def test_whitening_commutes_with_fractional_shift(noise_model):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, noise_model.n_samples))
    tau = np.array([0.3, -1.7, 5.25, 0.0])
    a = noise_model.whiten(shift(x, tau))
    b = shift(noise_model.whiten(x), tau)
    assert np.allclose(a, b, atol=1e-12)


def test_whitened_noise_is_white(noise_model):
    w = noise_model.whiten(noise_model.sample(20000, rng=2))
    C = w.T @ w / len(w)
    # projection onto the analysis band: eigenvalues ~1 (dof of them) and 0
    ev = np.sort(np.linalg.eigvalsh(C))[::-1][: noise_model.dof]
    assert abs(ev.mean() - 1) < 0.02
    assert noise_model.dof == noise_model.n_samples - 2


def test_shift_is_orthogonal_and_invertible(noise_model):
    rng = np.random.default_rng(3)
    x = noise_model.whiten(rng.standard_normal(noise_model.n_samples))
    y = shift(x, 2.37)
    assert np.isclose(x @ x, y @ y)
    assert np.allclose(shift(y, -2.37), x, atol=1e-12)


def test_from_records_recovers_spectrum(noise_model):
    est = StationaryNoise.from_records(noise_model.sample(4000, rng=4))
    r = est.lam[1:-1] / noise_model.lam[1:-1]
    assert abs(np.mean(r) - 1) < 0.02


def test_xcorr_matches_continuous_projection():
    rng = np.random.default_rng(5)
    n = 128
    Y = rng.standard_normal((3, n))
    B = np.linalg.qr(rng.standard_normal((n, 2)))[0].T
    lags, P = xcorr(Y, B, 4)
    for li, lag in enumerate(lags):
        x = aligned_projections(np.fft.rfft(Y, axis=1), np.fft.rfft(B, axis=1),
                                np.full(3, float(lag)), n)[0]
        direct = shift(Y, -lag) @ B.T
        assert np.allclose(x, P[:, :, li], atol=1e-10)
        # direct time-domain check (Nyquist bin removed by shift)
        Yn = np.fft.irfft(np.fft.rfft(Y, axis=1) * np.r_[np.ones(n // 2), 0], n=n, axis=1)
        assert np.allclose(shift(Yn, -lag) @ B.T, direct, atol=1e-10)


def test_projection_derivatives_match_finite_differences():
    rng = np.random.default_rng(6)
    n = 128
    Yh = np.fft.rfft(rng.standard_normal((2, n)), axis=1)
    Bh = np.fft.rfft(rng.standard_normal((3, n)), axis=1)
    tau = np.array([0.4, -2.2])
    x, dx, ddx = aligned_projections(Yh, Bh, tau, n, derivs=2)
    h = 1e-4
    xp = aligned_projections(Yh, Bh, tau + h, n)[0]
    xm = aligned_projections(Yh, Bh, tau - h, n)[0]
    assert np.allclose(dx, (xp - xm) / (2 * h), atol=1e-6)
    assert np.allclose(ddx, (xp - 2 * x + xm) / h ** 2, atol=1e-4)


def test_spectral_derivative_of_shift():
    n = 256
    t = np.arange(n)
    x = np.exp(-0.5 * ((t - 100) / 6.0) ** 2)
    h = 1e-5
    fd = (shift(x, h) - shift(x, -h)) / (2 * h)
    assert np.allclose(fd, -derivative(x), atol=1e-6)
