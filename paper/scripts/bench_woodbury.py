"""Benchmark: Woodbury/determinant-lemma likelihood vs dense Cholesky evaluation.

For each record length N we evaluate -2 log L(s, t0) for the non-stationary
model R(s, t0) = I + T_t0 V^T Sigma(s) V T_t0^T (whitened domain) by
(i) forming and Cholesky-factorising the dense N x N matrix (the historical
non-stationary filter of Fixsen et al. and Shank et al., O(N^3) per (s, t0))
and (ii) the K_b-dimensional formula of ManifoldFitter (O(K_b N) per record
and t0, after one FFT per record).  Both are checked to agree.
"""
import time

import numpy as np
from scipy.linalg import solve_triangular

from common import FIG, mpl, save_json, load_json
from empca import LowRankNoise, ManifoldFitter, PulseManifold, StationaryNoise
from empca.shifts import shift


class FixedSigma(LowRankNoise):
    def __init__(self, V, S0):
        super().__init__(n_modes=V.shape[0])
        self.V_, self._S0, self.floor = V, S0, 0.9

    def sigma_eig(self, s):
        s = np.atleast_1d(s)
        w, U = np.linalg.eigh(np.einsum("n,kl->nkl", s / 100.0, self._S0))
        return np.clip(w, -self.floor, None), U


def setup(N, Ks=4, Kn=8, seed=0):
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(N)
    lam = 1e-3 + 1.0 / (1 + (f / 0.01) ** 2)
    noise = StationaryNoise(lam, N)
    def band(k):
        x = rng.standard_normal((N, k))
        X = np.fft.rfft(x, axis=0) * noise.mask[:, None]
        return np.linalg.qr(np.fft.irfft(X, n=N, axis=0))[0].T
    Bs = band(Ks)
    u = np.linspace(0, 1, 400)
    a = np.stack([300 * u, 40 * u ** 2, 10 * u ** 3] + [0 * u] * (Ks - 3), 1)
    man = PulseManifold(n_knots=8).fit(a + 0.01 * rng.standard_normal(a.shape))
    V = band(Kn)
    S0 = np.diag(np.linspace(5, 0.5, Kn))
    fit = ManifoldFitter(noise, Bs, man, FixedSigma(V, S0))
    return noise, Bs, man, V, fit, rng


def dense_nll(noise, Bs, man, V, lr, yw, s, tau):
    N = noise.n_samples
    mu = shift(man.curve([s])[0] @ Bs, tau)
    TV = shift(V, tau)
    R = np.eye(N) + TV.T @ lr.sigma([s])[0] @ TV
    L = np.linalg.cholesky(R)
    z = solve_triangular(L, yw - mu, lower=True)
    return z @ z + 2 * np.sum(np.log(np.diag(L)))


def run():
    rows = []
    for N in (256, 512, 1024, 2048, 4096, 8192):
        noise, Bs, man, V, fit, rng = setup(N)
        n_rec = 2000
        yw = noise.whiten(noise.color(rng.standard_normal((n_rec, N))))
        raw = noise.color(yw)
        s = rng.uniform(50, 250, n_rec)
        tau = rng.uniform(-3, 3, n_rec)
        # fast: whitening FFT + K_b projections + K_b-dim algebra, per record
        t = time.perf_counter()
        fast = fit.nll(raw, s, tau)
        t_fast = (time.perf_counter() - t) / n_rec
        # fast, per additional (s, t0) evaluation once the record is in the frequency domain
        Yh = noise.whiten_rfft(raw)
        from empca.shifts import aligned_projections
        t = time.perf_counter()
        for _ in range(5):
            x = aligned_projections(Yh, fit.Bh, tau, N)[0]
            fit._f(x, s)
        t_eval = (time.perf_counter() - t) / (5 * n_rec)
        n_dense = {256: 50, 512: 30, 1024: 10, 2048: 4, 4096: 2, 8192: 1}[N]
        t = time.perf_counter()
        dens = np.array([dense_nll(noise, Bs, man, V, fit.lowrank, yw[j], s[j], tau[j])
                         for j in range(n_dense)])
        t_dense = (time.perf_counter() - t) / n_dense
        yy = np.sum(yw[:n_dense] ** 2, axis=1)
        rel = np.max(np.abs(dens - fast[:n_dense]) / np.abs(dens))
        rows.append(dict(N=N, t_fast=t_fast, t_eval=t_eval, t_dense=t_dense, max_rel_diff=rel,
                         K_b=fit.B.shape[0]))
        print(rows[-1], flush=True)
    save_json("woodbury_benchmark.json", rows)


def figure():
    rows = load_json("woodbury_benchmark.json")
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    N = np.array([r["N"] for r in rows])
    ax.loglog(N, [r["t_dense"] for r in rows], "o-", ms=3, color="#000000",
              label="dense Cholesky, per $(E,t_0)$")
    ax.loglog(N, [r["t_fast"] for r in rows], "s-", ms=3, color="#D55E00",
              label="Woodbury, per record (incl. FFT)")
    ax.loglog(N, [r["t_eval"] for r in rows], "^-", ms=3, color="#0072B2",
              label="Woodbury, per extra $(E,t_0)$")
    ref = rows[2]["t_dense"] * (N / N[2]) ** 3
    ax.loglog(N, ref, ":", color="grey", lw=0.8, label=r"$\propto N^3$")
    ax.set_xticks(N)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([str(n) for n in N])
    ax.set_xlabel("record length $N$")
    ax.set_ylabel("time per likelihood [s]")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/woodbury_timing.pdf")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["run", "figure"]
    if "run" in what:
        run()
    if "figure" in what:
        figure()
