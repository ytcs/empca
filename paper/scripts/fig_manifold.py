"""Figure: shift-invariant whitened PCA basis and the pulse manifold."""
import numpy as np

from common import FIG, mpl, simulate, trained, TRAIN_LINES

plt = mpl()
M = trained("morph")
pca, man, noise = M["pca"], M["man"], M["noise"]
D = simulate("morph")
E = D["train_E"]
t_us = (np.arange(noise.n_samples) - 256) / 200e3 * 1e6

fig = plt.figure(figsize=(7.2, 2.3))
ax1 = fig.add_subplot(1, 4, 1)
cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
for k, b in enumerate(pca.basis_):
    ax1.plot(t_us, b / np.abs(b).max() - 1.2 * k, color=cols[k % 6], lw=0.8)
ax1.set_xlim(-200, 1500)
ax1.set_yticks([])
ax1.set_xlabel(r"$t - t_0$ [$\mu$s]")
ax1.set_title("(a) whitened basis $B$", fontsize=8)

ax2 = fig.add_subplot(1, 4, 2)
for k, b in enumerate(pca.templates()):
    ax2.plot(t_us, b / np.abs(b).max() - 1.2 * k, color=cols[k % 6], lw=0.8)
ax2.set_xlim(-200, 1500)
ax2.set_yticks([])
ax2.set_xlabel(r"$t - t_0$ [$\mu$s]")
ax2.set_title(r"(b) coloured, $R_0^{1/2}B$", fontsize=8)

A = pca.coeffs_
Z = A @ man.rotation_.T
grid = np.linspace(-200, man.s_max_ * 1.02, 600)
C = man.curve(grid) @ man.rotation_.T
rng = np.random.default_rng(0)
idx = rng.choice(len(A), 6000, replace=False)
for j, (i1, i2) in enumerate(((0, 1), (1, 2))):
    ax = fig.add_subplot(1, 4, 3 + j)
    sc = ax.scatter(Z[idx, i1], Z[idx, i2], c=E[idx] / 1e3, s=0.6, cmap="viridis", rasterized=True)
    ax.plot(C[:, i1], C[:, i2], color="#D55E00", lw=0.9)
    ax.set_xlabel(f"$z_{i1 + 1}$")
    ax.set_ylabel(f"$z_{i2 + 1}$")
    ax.set_title("(c) manifold" if j == 0 else "(d) manifold, transverse", fontsize=8)
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
cb.set_label("E [keV]")
fig.tight_layout(w_pad=0.5)
fig.savefig(f"{FIG}/manifold.pdf", dpi=300)

print("SIPCA iterations", len(pca.objective_), "objective", pca.objective_)
print("singular values", pca.singular_values_)
print("manifold lambda", man.lambda_, "s_max", man.s_max_, "mean dist^2", np.mean(man.distance_ ** 2))
for e in TRAIN_LINES:
    print(e, "median s", np.median(M["s"][E == e]))
