"""Label-free sampling-phase correction of arrival times at high statistics.

For mono-energetic sets of 10^5 simulated records (no morphology variation),
the PhaseEqualizer is learned on one half and applied to the other, for
increasing numbers of learning records.  Reported: arrival-time jitter before
and after the correction, and the learned-model CRB.
"""
import numpy as np

from common import FIG, COLORS, load_json, mpl, save_json, trained
from empca import ManifoldFitter, PhaseEqualizer
from empca.simulate import TESParameters, TESSimulator

ENERGIES = (2200., 6500., 11000.)
N_EACH = 100000
N_LEARN = (1000, 3000, 10000, 30000, 50000)


def rsd(v):
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def run():
    M = trained("nomorph")
    f = ManifoldFitter(M["noise"], M["pca"].basis_, M["man"], M["lr_lines"])
    sim = TESSimulator(TESParameters(eta_scale=0.0), seed=77)
    out = []
    for e in ENERGIES:
        t0 = 256 + sim.rng.uniform(-2, 2, N_EACH)
        X = sim.simulate(np.full(N_EACH, e), None, t0)
        tau_a, a = M["pca"].transform(X)
        s1, t1, _ = M["fit0"].fit(X, M["man"].project(a), tau_a)
        s2, tau, _ = f.fit(X, s1, t1)
        err = tau - (t0 - 256)
        test = np.arange(N_EACH) >= N_EACH // 2
        g = np.zeros(N_EACH, int)
        row = dict(E=e, jitter=rsd(err[test]), crb=float(f.crb(np.median(s2))[1]), after={})
        for n in N_LEARN:
            learn = np.where(~test)[0][:n]
            gl = np.full(N_EACH, -1)
            gl[learn] = 0
            eq = PhaseEqualizer(3).fit(tau, gl)
            tc = eq.transform(tau[test], g[test])
            row["after"][str(n)] = rsd(tc - (t0[test] - 256))
        out.append(row)
        print(e, round(row["jitter"] * 5e3, 1), {k: round(v * 5e3, 1) for k, v in row["after"].items()},
              "CRB", round(row["crb"] * 5e3, 1), flush=True)
    save_json("timing_equalize.json", out)


def figure():
    rows = load_json("timing_equalize.json")
    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cols = ["#56B4E9", "#E69F00", "#D55E00"]
    for r, c in zip(rows, cols):
        n = np.array([int(k) for k in r["after"]])
        ax.plot(n, np.array(list(r["after"].values())) * 5e3, "o-", ms=3, color=c,
                label=f"{r['E'] / 1e3:g} keV, equalised")
        ax.axhline(r["jitter"] * 5e3, color=c, ls="--", lw=0.8)
        ax.axhline(r["crb"] * 5e3, color=c, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([3, 5, 10, 20, 50])
    ax.set_xlabel("records used to learn the correction")
    ax.set_ylabel(r"jitter $\sigma_{t_0}$ [ns]")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("dashed: uncorrected; dotted: CRB", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/timing_equalize.pdf")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["run", "figure"]
    if "run" in what:
        run()
    if "figure" in what:
        figure()
