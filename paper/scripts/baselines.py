"""Prior-art baselines, implemented as faithfully as practical.

All baselines start from the classical optimal filter (OF, 5.9 keV template,
arrival-time search) and use the *training* lines with their known identities
(as a practitioner would after a calibration run).  They are evaluated on the
same held-out test lines as the proposed method.

* ``phase``   -- OF + empirical arrival-phase correction (NIST practice,
  Fowler et al. 2016): per-line Fourier-series fit of the filtered value
  against the sub-sample phase, interpolated in energy.
* ``decor``   -- OF + phase + linear decorrelation against the leading
  noise-weighted SVD components of the OF residuals (NIST-style "projectors"),
  per line and interpolated in energy.  This is the strongest
  practitioner-style correction we could construct.
* ``secant``  -- stationary-noise fit along the piecewise-linear curve through
  the per-line mean pulses: the discrete form of the tangent filter of
  Fowler et al. 2017 / the geometric estimator of Fixsen et al. 2014 with
  stationary noise.
* ``covar``   -- the same curve with an energy-dependent full covariance per
  line (Ledoit-Wolf shrinkage), linearly interpolated between the bracketing
  lines: the non-stationary filter of Fixsen et al. 2004/2014 as implemented
  in SIRENA's covariance-matrix methods.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from empca.shifts import shift_rfft


def harmonics(phi, n=3):
    return np.stack([f(2 * np.pi * k * phi) for k in range(1, n + 1) for f in (np.cos, np.sin)], 1)


def aligned(noise, X, t0):
    n = noise.n_samples
    return np.fft.irfft(shift_rfft(noise.whiten_rfft(X), -t0, n), n=n, axis=1)


def ledoit_wolf(R):
    """Ledoit-Wolf shrinkage of the covariance of centred, whitened rows R towards I.

    In the whitened domain the baseline noise covariance is the identity, which is
    therefore the natural shrinkage target (a scaled identity would be inflated by
    the few large pulse-correlated eigenvalues)."""
    n, p = R.shape
    S = R.T @ R / n
    d2 = np.sum((S - np.eye(p)) ** 2)
    b2 = (np.sum(np.sum(R * R, axis=1) ** 2) - n * np.sum(S * S)) / n ** 2
    a = min(max(b2, 0.0), d2) / d2
    return a * np.eye(p) + (1 - a) * S


def _interp_coef(knots_x, coefs, x):
    """Linear interpolation of per-line coefficient vectors in x (clamped)."""
    order = np.argsort(knots_x)
    kx, kc = np.asarray(knots_x)[order], np.asarray(coefs)[order]
    return np.stack([np.interp(x, kx, kc[:, j]) for j in range(kc.shape[1])], 1)


class PracticeBaselines:
    """Fit on the training set, apply to the test set."""

    def __init__(self, of, noise, lines, n_proj=8):
        self.of, self.noise, self.lines, self.n_proj = of, noise, np.asarray(lines), n_proj

    def fit(self, X, lab):
        A, tau = self.of.transform(X)
        self.A_med = np.array([np.median(A[lab == i]) for i in range(len(self.lines))])
        # energy scale of the OF (for choosing segments / interpolation knots)
        self.calA = PchipInterpolator(np.r_[0, self.A_med], np.r_[0, self.lines], extrapolate=True)
        use = lab >= 0
        # --- phase correction
        cp = []
        for i in range(len(self.lines)):
            m = lab == i
            H = harmonics(np.mod(tau[m], 1))
            y = A[m] / self.A_med[i] - 1
            cp.append(np.linalg.lstsq(np.c_[np.ones(m.sum()), H], y, rcond=None)[0][1:])
        self.c_phase = np.array(cp)
        # --- residual projectors (noise-weighted SVD of OF residuals, pooled)
        hw = self.of._b[0] * np.sqrt(self.of.norm2_)
        self.hw = hw
        Xa = aligned(self.noise, X[use], tau[use])
        Rr = Xa - A[use, None] * hw
        # practitioner-style outlier cut (pile-up, glitches) on the residual norm
        rn = np.sum(Rr * Rr, axis=1)
        med = np.median(rn)
        clean = rn < med + 5 * 1.4826 * np.median(np.abs(rn - med))
        w, U = np.linalg.eigh(Rr[clean].T @ Rr[clean] / clean.sum())
        self.V = U[:, np.argsort(w)[::-1][: self.n_proj]].T
        F = self._features(Xa, A[use], tau[use])
        cd = []
        lab_u = lab[use]
        for i in range(len(self.lines)):
            m = (lab_u == i) & clean
            Fm = F[m] - F[m].mean(0)
            y = A[use][m] - A[use][m].mean()
            keep = np.ones(m.sum(), bool)
            for _ in range(5):  # iteratively trimmed least squares (robust to pile-up)
                beta = np.linalg.lstsq(Fm[keep], y[keep], rcond=None)[0]
                r = y - Fm @ beta
                sd = 1.4826 * np.median(np.abs(r - np.median(r)))
                keep = np.abs(r - np.median(r)) < 3 * sd
            cd.append(beta)
        self.c_decor = np.array(cd)
        # --- piecewise-linear pulse curve and per-line covariances (whitened domain)
        mus, Ws = [np.zeros(self.noise.n_samples)], [np.eye(self.noise.n_samples)]
        for i in range(len(self.lines)):
            m = lab_u == i
            mu = Xa[m].mean(0)
            mus.append(mu)
            Ws.append(np.linalg.inv(ledoit_wolf(Xa[m] - mu)))
        self.E_knots = np.r_[0.0, self.lines]
        self.mus, self.Ws = np.array(mus), Ws
        return self

    def _features(self, Xa, A, tau):
        z = (Xa - A[:, None] * self.hw) @ self.V.T
        return np.c_[z, harmonics(np.mod(tau, 1))]

    def transform(self, X):
        A, tau = self.of.transform(X)
        out = {}
        corr = np.sum(harmonics(np.mod(tau, 1)) * _interp_coef(self.A_med, self.c_phase, A), 1)
        out["phase"] = A / (1 + corr)
        Xa = aligned(self.noise, X, tau)
        F = self._features(Xa, A, tau)
        out["decor"] = A - np.sum(F * _interp_coef(self.A_med, self.c_decor, A), 1)
        E0 = self.calA(A)
        seg = np.clip(np.searchsorted(self.E_knots, E0) - 1, 0, len(self.E_knots) - 2)
        for name, use_cov in (("secant", False), ("covar", True)):
            est = np.empty(len(X))
            for k in range(len(self.E_knots) - 1):
                m = seg == k
                if not m.any():
                    continue
                Ea, Eb = self.E_knots[k], self.E_knots[k + 1]
                d = self.mus[k + 1] - self.mus[k]
                r = Xa[m] - self.mus[k]
                if use_cov:
                    wgt = np.clip((E0[m] - Ea) / (Eb - Ea), 0, 1)
                    ga, gb = self.Ws[k] @ d, self.Ws[k + 1] @ d
                    num = (1 - wgt) * (r @ ga) + wgt * (r @ gb)
                    den = (1 - wgt) * (d @ ga) + wgt * (d @ gb)
                else:
                    num, den = r @ d, d @ d
                est[m] = Ea + (Eb - Ea) * num / den
            out[name] = est
        return out, tau
