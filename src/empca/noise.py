"""Stationary baseline noise model and shift-commuting spectral whitening.

For a record of ``N`` samples the stationary noise covariance is approximated by
a circulant matrix ``R0 = F^H diag(lam) F``.  Its symmetric inverse square root
``W = F^H diag(lam^{-1/2}) F`` whitens the noise *and commutes with every
(circular, possibly fractional) time shift*, which a Cholesky factor of the
Toeplitz covariance does not.  This is what makes shift-invariant processing in
the whitened domain exact.

The DC bin (baseline offset) and the Nyquist bin are excluded by default; all
whitened vectors then live in an ``N - 2`` dimensional subspace on which the
noise is exactly white with unit variance.
"""
import numpy as np

__all__ = ["StationaryNoise", "rfft_weights"]


def rfft_weights(n):
    """Multiplicity of each rfft bin in a full-spectrum sum (1 for DC/Nyquist, else 2)."""
    w = np.full(n // 2 + 1, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return w


class StationaryNoise:
    """Circulant (stationary) noise model with spectral whitening.

    Parameters
    ----------
    lam : array_like, shape (N//2 + 1,)
        Eigenvalues of the circulant covariance, ``lam_k = E|Y_k|^2 / N`` with
        ``Y = numpy.fft.rfft(noise)``.
    n_samples : int
        Record length ``N``.
    mask : array_like of bool, optional
        Frequency bins used in the analysis.  Default: all but DC and Nyquist.
    """

    def __init__(self, lam, n_samples, mask=None):
        self.n_samples = int(n_samples)
        self.lam = np.asarray(lam, float)
        if self.lam.shape != (self.n_samples // 2 + 1,):
            raise ValueError("lam must have length n_samples // 2 + 1")
        if mask is None:
            mask = np.ones_like(self.lam, bool)
            mask[0] = False
            if self.n_samples % 2 == 0:
                mask[-1] = False
        self.mask = np.asarray(mask, bool)
        self._wgain = np.where(self.mask, 1.0 / np.sqrt(np.where(self.lam > 0, self.lam, np.inf)), 0.0)

    # ---------------------------------------------------------- construction
    @classmethod
    def from_records(cls, records, mask=None, smooth=0):
        """Estimate the noise spectrum from pulse-free records (rows).

        ``smooth`` > 0 applies a running mean of that half-width (in bins) to
        the periodogram, trading frequency resolution for variance.
        """
        records = np.atleast_2d(np.asarray(records, float))
        n = records.shape[1]
        records = records - records.mean(axis=1, keepdims=True)
        lam = np.mean(np.abs(np.fft.rfft(records, axis=1)) ** 2, axis=0) / n
        if smooth:
            k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
            padded = np.pad(lam, smooth, mode="edge")
            lam = np.convolve(padded, k, mode="valid")
        return cls(lam, n, mask)

    @classmethod
    def from_psd(cls, psd, fs, n_samples, mask=None):
        """Build from a one-sided PSD [units^2/Hz] sampled on ``rfftfreq(n, 1/fs)``."""
        psd = np.asarray(psd, float)
        lam = psd * fs / 2.0
        if n_samples % 2 == 0:
            lam = lam.copy()
            lam[[0, -1]] *= 2.0
        else:
            lam = lam.copy()
            lam[0] *= 2.0
        return cls(lam, n_samples, mask)

    # ---------------------------------------------------------------- props
    @property
    def n_freq(self):
        return self.lam.size

    @property
    def dof(self):
        """Dimension of the whitened signal space."""
        return int(np.sum(rfft_weights(self.n_samples)[self.mask]))

    def psd(self, fs):
        """One-sided PSD [units^2/Hz] on ``numpy.fft.rfftfreq(N, 1/fs)``."""
        p = 2.0 * self.lam / fs
        p[0] /= 2.0
        if self.n_samples % 2 == 0:
            p[-1] /= 2.0
        return p

    # ------------------------------------------------------------ operators
    def whiten_rfft(self, x):
        """rfft of the whitened signal (masked bins set to zero)."""
        return np.fft.rfft(x, axis=-1) * self._wgain

    def whiten(self, x):
        """Apply ``R0^{-1/2}`` (projected on the analysis band)."""
        return np.fft.irfft(self.whiten_rfft(x), n=self.n_samples, axis=-1)

    def color(self, xw):
        """Apply ``R0^{1/2}`` (inverse of :meth:`whiten` on the analysis band)."""
        g = np.where(self.mask, np.sqrt(self.lam), 0.0)
        return np.fft.irfft(np.fft.rfft(xw, axis=-1) * g, n=self.n_samples, axis=-1)

    def covariance(self):
        """Dense circulant covariance (for testing / small problems)."""
        n = self.n_samples
        acf = np.fft.irfft(self.lam, n=n)
        idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
        return acf[idx]

    def sample(self, n_records, rng=None):
        """Draw Gaussian noise records with this spectrum."""
        rng = np.random.default_rng(rng)
        n = self.n_samples
        white = rng.standard_normal((n_records, n))
        return np.fft.irfft(np.fft.rfft(white, axis=1) * np.sqrt(self.lam), n=n, axis=1)
