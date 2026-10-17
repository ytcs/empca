"""Figure: low-rank structure of the pulse-correlated noise."""
import numpy as np

from common import FIG, mpl, simulate, trained, TRAIN_LINES
from empca.shifts import shift_rfft

plt = mpl()
M = trained("morph")
pca, man, noise, s, t0 = M["pca"], M["man"], M["noise"], M["s"], M["t0"]
D = simulate("morph")
n = noise.n_samples
Xa = np.fft.irfft(shift_rfft(noise.whiten_rfft(D["train"]), -t0, n), n=n, axis=1)
groups = M["groups"]

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
# (a) eigenvalue spectra of the residual covariance at three lines
for g, col in zip((0, 3, 6), ("#56B4E9", "#E69F00", "#D55E00")):
    m = groups == g
    r = Xa[m] - Xa[m].mean(0)
    ev = np.sort(np.linalg.eigvalsh(r.T @ r / m.sum()))[::-1]
    e_line = np.median(D["train_E"][m])
    ax[0].semilogy(np.arange(1, 41), ev[:40], marker=".", ms=3, color=col,
                   label=f"{e_line / 1e3:g} keV line")
    gam = n / m.sum()
    ax[0].axhline((1 + np.sqrt(gam)) ** 2, color="grey", lw=0.5, ls=":")
ax[0].set_xlabel("eigenvalue index")
ax[0].set_ylabel("eigenvalue (whitened)")
ax[0].legend(frameon=False)
ax[0].set_title("(a) residual covariance spectra", fontsize=8)

# (b) leading modes in the raw domain
K = pca.basis_.shape[0]
V = M["lr_lines"].V_[K:]  # residual modes beyond the signal subspace
t_us = (np.arange(n) - 256) / 200e3 * 1e6
cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
for k in range(4):
    v = noise.color(V[k])
    ax[1].plot(t_us, v / np.abs(v).max() - 1.2 * k, color=cols[k], lw=0.8)
ax[1].set_xlim(-200, 1000)
ax[1].set_yticks([])
ax[1].set_xlabel(r"$t - t_0$ [$\mu$s]")
ax[1].set_title("(b) leading noise modes $R_0^{1/2}V$", fontsize=8)

# (c) energy dependence of the excess covariance
lr = M["lr_lines"]
sg = np.linspace(0, lr.s_knots_[-1], 300)
S = lr.sigma(sg)
for k in range(4):
    j = K + k
    ax[2].plot(sg, S[:, j, j], color=cols[k], label=f"mode {k + 1}")
    ax[2].plot(lr.s_knots_, lr.sigma_knots_[:, j, j], "o", ms=2.5, color=cols[k])
ax[2].plot(sg, np.linalg.eigvalsh(S[:, :K, :K])[:, 0], color="grey", ls=":",
           label="signal subspace, min. eig.")
ax[2].set_yscale("symlog", linthresh=1)
ax[2].set_xlabel("manifold coordinate $s$")
ax[2].set_ylabel("excess variance")
ax[2].legend(frameon=False, fontsize=6)
ax[2].set_title(r"(c) $\Sigma(s)$ (line-aware)", fontsize=8)
fig.tight_layout(w_pad=0.6)
fig.savefig(f"{FIG}/covariance.pdf")

# correlations of the leading line-residual modes with true eta and sampling phase
for g in (3, 6):
    m = groups == g
    r = Xa[m] - Xa[m].mean(0)
    z = r @ V[:4].T
    eta = D["train_eta"][m]
    ph = D["train_t0"][m]
    for k in range(4):
        c = [np.corrcoef(z[:, k], v)[0, 1] for v in (eta, np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph))]
        print("line", g, "mode", k, "corr(eta, cos2pi t0, sin2pi t0)", np.round(c, 2))
print("pooled eigenvalues", np.round(lr.pooled_eigenvalues_, 1))
