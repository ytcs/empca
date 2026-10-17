"""Classical optimal filter (OF) with arrival-time fit, as the reference baseline."""
import numpy as np

from .shifts import xcorr, aligned_projections, shift_rfft
from .sipca import refine_peak

__all__ = ["OptimalFilter"]


class OptimalFilter:
    """Frequency-domain optimal filter with sub-sample arrival-time search.

    The amplitude estimator is ``A(tau) = <h_w, T_{-tau} y_w> / ||h_w||^2`` with
    ``h_w`` the whitened template; ``tau`` maximises ``A`` (the usual OF with
    arrival-time correction).  With ``template=None`` the template is the
    iteratively aligned average of the records passed to :meth:`fit` (e.g.
    the events of one calibration line).
    """

    def __init__(self, max_shift=16, n_align=3):
        self.max_shift = max_shift
        self.n_align = n_align

    def fit(self, traces, noise, template=None):
        self.noise_ = noise
        n = noise.n_samples
        if template is None:
            traces = np.atleast_2d(np.asarray(traces, float))
            h = traces.mean(axis=0)
            for _ in range(self.n_align):
                self._set_template(h)
                tau, _ = self._tau(noise.whiten_rfft(traces))
                tau -= np.median(tau)
                h = np.fft.irfft(shift_rfft(np.fft.rfft(traces, axis=1), -tau, n), n=n, axis=1).mean(axis=0)
        else:
            h = np.asarray(template, float)
        self._set_template(h)
        return self

    def _set_template(self, h):
        self.template_ = h - h.mean()
        hw = self.noise_.whiten(self.template_)
        self.norm2_ = float(hw @ hw)
        self._b = (hw / np.sqrt(self.norm2_))[None, :]
        self._bh = np.fft.rfft(self._b, axis=1)

    def _tau(self, Yh):
        lags, P = xcorr(Yh, self._b, self.max_shift)
        tau0 = lags[np.argmax(P[:, 0, :], axis=1)].astype(float)
        return refine_peak(Yh, self._bh, tau0, self.noise_.n_samples, bound=self.max_shift + 1)

    def transform(self, traces):
        """Return ``(amplitude, t0)``; amplitude in units of the template."""
        Yh = self.noise_.whiten_rfft(np.atleast_2d(np.asarray(traces, float)))
        tau, _ = self._tau(Yh)
        x = aligned_projections(Yh, self._bh, tau, self.noise_.n_samples)[0][:, 0]
        return x / np.sqrt(self.norm2_), tau

    @property
    def resolution(self):
        """Baseline amplitude resolution (1 sigma), in template units."""
        return 1.0 / np.sqrt(self.norm2_)
