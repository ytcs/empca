"""Figure: low-energy behaviour -- rectification bias of the coefficient norm
(Rician/chi bias of Yu 2019) versus the unbiased manifold coordinate.  Each
estimator is calibrated with a quadratic through the origin fitted at 80-320 eV."""
import numpy as np

from common import FIG, mpl, save_json, simulate, trained
from empca import OptimalFilter
from empca.simulate import TESParameters, TESSimulator

plt = mpl()
M = trained("morph")
pca, man, fit0, noise = M["pca"], M["man"], M["fit0"], M["noise"]
D = simulate("morph")
sim = TESSimulator(TESParameters(eta_scale=0.02), seed=21)
energies = np.array([0., 5., 10., 20., 40., 80., 160., 320.])
n_each = 1500
E = np.repeat(energies, n_each)
X = sim.simulate(E, sim.rng.uniform(-1, 1, E.size), 256 + sim.rng.uniform(-2, 2, E.size))

of = OptimalFilter(max_shift=8).fit(D["train"][D["train_E"] == 1500.], noise)

# fixed arrival-time search window for every estimator (low-SNR records)
tau, a = pca.transform(X)
s_proj = man.project(a)
s_ml, _, _ = fit0.fit(X, s_proj, tau)
A, _ = of.transform(X)
raw = {"PCA coefficient norm": np.linalg.norm(a, axis=1), "Optimal filter": A,
       "Manifold ML (this work)": s_ml}
# each estimator is calibrated with a quadratic through the origin fitted to the
# high-SNR points (80-320 eV); the TES is slightly non-linear even there
def calib(v):
    e_ref = energies[energies >= 80]
    m = np.array([np.mean(v[E == e]) for e in e_ref])
    g = np.linalg.lstsq(np.vstack([m, m ** 2]).T, e_ref, rcond=None)[0]
    return g[0] * v + g[1] * v * np.abs(v)
est = {k: calib(v) for k, v in raw.items()}
cols = {"PCA coefficient norm": "#999999", "Optimal filter": "#000000",
        "Manifold ML (this work)": "#D55E00"}

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
out = {}
for k, v in est.items():
    mu = np.array([np.mean(v[E == e]) for e in energies])
    sd = np.array([np.std(v[E == e]) for e in energies])
    out[k] = dict(mean=mu.tolist(), std=sd.tolist())
    ax[0].errorbar(energies + 1, mu - energies, sd / np.sqrt(n_each), marker="o", ms=3,
                   color=cols[k], label=k, capsize=0)
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xscale("log")
ax[0].set_xlabel("true energy + 1 [eV]")
ax[0].set_ylabel(r"bias $\langle\hat E\rangle - E$ [eV]")
ax[0].legend(frameon=False)
ax[0].set_title("(a) bias at low energy", fontsize=8)
bins = np.linspace(-4, 5, 61)
for k, v in est.items():
    ax[1].hist(v[E == 0], bins=bins, histtype="step", color=cols[k], label=k, density=True)
ax[1].set_xlabel(r"$\hat E$ for pure noise records [eV]")
ax[1].set_ylabel("density")
ax[1].set_title("(b) zero-energy response", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/lowenergy.pdf")
save_json("lowenergy.json", dict(energies=energies.tolist(), estimators=out))
for k, v in out.items():
    print(k, "bias", np.round(np.array(v["mean"]) - energies, 2), "std", np.round(v["std"], 2))
