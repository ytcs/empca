"""Band-limited (Fourier) time shifts and phase-coherent cross-correlation.

Conventions
-----------
``shift(x, tau)`` delays ``x`` by ``tau`` samples (``tau`` may be fractional):
``(T_tau x)(t) = x(t - tau)``, implemented as multiplication of the rfft by
``exp(-2 pi i k tau / N)``.  The Nyquist bin is zeroed, which makes ``T_tau``
exactly orthogonal on the remaining band for any real ``tau``.

For a basis ``B`` (rows ``b_k``) and a record ``y`` the *aligned projections*
are ``x_k(tau) = <b_k, T_{-tau} y>``.  They are evaluated either for all
integer lags at once with one inverse FFT (:func:`xcorr`) or for arbitrary
continuous ``tau`` together with their first two derivatives
(:func:`aligned_projections`).
"""
import numpy as np

from .noise import rfft_weights

__all__ = ["shift", "shift_rfft", "xcorr", "aligned_projections", "derivative"]


def _phase(n, tau):
    k = np.arange(n // 2 + 1)
    ph = np.exp(-2j * np.pi * np.multiply.outer(np.asarray(tau, float), k) / n)
    if n % 2 == 0:
        ph[..., -1] = 0.0
    return ph


def shift_rfft(X, tau, n):
    """Shift in the rfft domain: ``X`` (..., n//2+1), ``tau`` broadcastable to X[..., 0]."""
    return X * _phase(n, tau)


def shift(x, tau):
    """Delay real signals ``x`` (..., N) by ``tau`` samples (broadcast over leading axes)."""
    x = np.asarray(x, float)
    n = x.shape[-1]
    return np.fft.irfft(shift_rfft(np.fft.rfft(x, axis=-1), tau, n), n=n, axis=-1)


def derivative(x, order=1):
    """Spectral time derivative d^m x / dt^m (per sample), Nyquist bin zeroed."""
    x = np.asarray(x, float)
    n = x.shape[-1]
    k = np.arange(n // 2 + 1)
    f = (2j * np.pi * k / n) ** order
    if n % 2 == 0:
        f[-1] = 0.0
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * f, n=n, axis=-1)


def xcorr(Y, B, max_lag, batch=2048):
    """Aligned projections for all integer lags ``|tau| <= max_lag``.

    Parameters
    ----------
    Y : ndarray (n, N) records (time domain) or rfft (n, N//2+1) if complex.
    B : ndarray (K, N) basis (time domain).
    max_lag : int

    Returns
    -------
    lags : ndarray (2*max_lag+1,)
    P : ndarray (n, K, 2*max_lag+1) with ``P[j, k, l] = <b_k, T_{-lags[l]} y_j>``.
    """
    B = np.atleast_2d(B)
    n = B.shape[1]
    Bh = np.conj(np.fft.rfft(B, axis=1))
    if n % 2 == 0:
        Bh[:, -1] = 0.0  # same band as shift(): Nyquist bin excluded
    lags = np.arange(-max_lag, max_lag + 1)
    Y = np.atleast_2d(Y)
    out = np.empty((Y.shape[0], B.shape[0], lags.size))
    for lo in range(0, Y.shape[0], batch):
        Yb = Y[lo:lo + batch]
        Yh = Yb if np.iscomplexobj(Yb) else np.fft.rfft(Yb, axis=1)
        c = np.fft.irfft(Yh[:, None, :] * Bh[None], n=n, axis=-1)
        out[lo:lo + batch] = c[..., lags % n]
    return lags, out


def aligned_projections(Yh, Bh, tau, n, derivs=0):
    """``x_k(tau) = <b_k, T_{-tau} y>`` at continuous ``tau`` (one per record).

    Parameters
    ----------
    Yh : ndarray (n_rec, F) rfft of records.
    Bh : ndarray (K, F) rfft of basis vectors.
    tau : ndarray (n_rec,)
    n : int, record length.
    derivs : int, number of tau-derivatives to return as well (0, 1 or 2).

    Returns
    -------
    list of ndarrays (n_rec, K): [x, dx/dtau, d2x/dtau2][:derivs+1]
    """
    w = rfft_weights(n) / n
    k = np.arange(n // 2 + 1)
    # T_{-tau} multiplies by exp(+2 pi i k tau / N)
    ph = np.exp(2j * np.pi * np.multiply.outer(np.asarray(tau, float), k) / n)
    if n % 2 == 0:
        ph[:, -1] = 0.0
    Z = Yh * ph
    G = (np.conj(Bh) * w).T  # (F, K)
    out = [np.real(Z @ G)]
    if derivs >= 1:
        iw = 2j * np.pi * k / n
        out.append(np.real((Z * iw) @ G))
        if derivs >= 2:
            out.append(np.real((Z * iw ** 2) @ G))
    return out
