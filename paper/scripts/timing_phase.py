"""Diagnostic: how much of the timing jitter is a deterministic function of the
sub-sample arrival phase (residual aliasing of the pulse onset)?

For each test line we regress the timing error of the line-aware Woodbury ML
on the first three harmonics of the *true* sub-sample phase and report the
robust jitter of the remainder.  This is an oracle (it uses the true t0); it
quantifies the phase bias, it is not an estimator.
"""
import numpy as np

from common import TEST_LINES, save_json, simulate, trained
from empca import ManifoldFitter


def rsd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


out = {}
for sc in ("morph", "nomorph"):
    M, D = trained(sc), simulate(sc)
    X, E, t0 = D["test"], D["test_E"], D["test_t0"] - 256
    tau_a, a = M["pca"].transform(X)
    s1, tau1, _ = M["fit0"].fit(X, M["man"].project(a), tau_a)
    s2, tau2, _ = ManifoldFitter(M["noise"], M["pca"].basis_, M["man"], M["lr_lines"]).fit(X, s1, tau1)
    rows = []
    for e in TEST_LINES:
        m = E == e
        d = tau2[m] - t0[m]
        ph = np.mod(t0[m], 1.0)
        A = np.vstack([np.ones(m.sum())] + [f(2 * np.pi * k * ph) for k in (1, 2, 3)
                                            for f in (np.cos, np.sin)]).T
        res = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
        rows.append(dict(E=e, jitter=rsd(d), jitter_phase_removed=rsd(res)))
    out[sc] = rows
    print(sc, [(r["E"], round(r["jitter"] * 5e3, 1), round(r["jitter_phase_removed"] * 5e3, 1)) for r in rows])
save_json("timing_phase.json", out)
