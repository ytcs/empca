"""Main benchmark: energy and timing resolution of all estimators on held-out data.

All models are learned on an unlabelled training spectrum (lines at
TRAIN_LINES on a flat continuum) and evaluated on an independent test spectrum
whose lines (TEST_LINES) lie *between* the training lines.  Resolutions are
converted to eV with the local gain measured on the test continuum, which is
identical for every estimator and removes calibration-interpolation effects.
"""
import time

import numpy as np

from common import (COLORS, FIG, TRAIN_LINES, TEST_LINES, evaluate, load_json, mpl,
                    save_json, simulate)
from empca import (LowRankNoise, ManifoldFitter, OptimalFilter, PulseManifold,
                   ShiftInvariantPCA, StationaryNoise, find_clusters)
from empca.calibration import FWHM
from empca.shifts import shift_rfft
from baselines import PracticeBaselines

MAX_SHIFT, SNR_MIN = 8, 30.0


def aligned(noise, X, t0):
    n = noise.n_samples
    return np.fft.irfft(shift_rfft(noise.whiten_rfft(X), -t0, n), n=n, axis=1)


def stage_ab(D, noise, K):
    pca = ShiftInvariantPCA(K, n_align=1, max_shift=MAX_SHIFT, snr_min=SNR_MIN).fit(D["train"], noise)
    man = PulseManifold().fit(pca.coeffs_)
    fit0 = ManifoldFitter(noise, pca.basis_, man)
    s, t0, _ = fit0.fit(D["train"], man.s_, pca.t0_)
    # second pass (as in PulseProcessor): clusters enter the curve only via their means
    groups, _ = find_clusters(s, len(TRAIN_LINES))
    man = PulseManifold().fit(pca.coeffs_, groups=groups)
    fit0 = ManifoldFitter(noise, pca.basis_, man)
    s, t0, _ = fit0.fit(D["train"], man.s_, pca.t0_)
    return pca, man, fit0, s, t0


def covariances(D, noise, pca, man, s, t0, Kn):
    Xa = aligned(noise, D["train"], t0)
    groups, _ = find_clusters(s, len(TRAIN_LINES))
    lr_lines = LowRankNoise(Kn).fit(Xa, s, groups=groups, include=pca.basis_)
    lr_curve = LowRankNoise(Kn).fit(Xa - man.curve(s) @ pca.basis_, s, include=pca.basis_)
    return lr_lines, lr_curve, groups


def run_scenario(scenario, K=4, Kn=8):
    D = simulate(scenario)
    noise = StationaryNoise.from_records(D["noise"])
    E, lab, t0_true = D["test_E"], D["test_lab"], D["test_t0"] - 256
    X = D["test"]
    res, timing = {}, {}

    t = time.time()
    of = OptimalFilter(max_shift=MAX_SHIFT).fit(D["train"][D["train_E"] == 5900.], noise)
    A, tauA = of.transform(X)
    timing["OF"] = time.time() - t
    res["OF"] = evaluate(A, E, lab, TEST_LINES, tauA, t0_true)

    t = time.time()
    pca, man, fit0, s_tr, t0_tr = stage_ab(D, noise, K)
    timing["learn_AB"] = time.time() - t
    t = time.time()
    tau_a, a = pca.transform(X)
    s_proj = man.project(a)
    timing["SIPCA+projection"] = time.time() - t
    res["PCA norm"] = evaluate(np.linalg.norm(a, axis=1), E, lab, TEST_LINES, tau_a, t0_true)
    res["Manifold projection"] = evaluate(s_proj, E, lab, TEST_LINES, tau_a, t0_true)
    t = time.time()
    s1, tau1, _ = fit0.fit(X, s_proj, tau_a)
    timing["ML stationary"] = time.time() - t
    res["ML stationary"] = evaluate(s1, E, lab, TEST_LINES, tau1, t0_true)

    t = time.time()
    lr_lines, lr_curve, groups = covariances(D, noise, pca, man, s_tr, t0_tr, Kn)
    timing["learn_C"] = time.time() - t
    out = {}
    for name, lr in (("ML Woodbury (curve)", lr_curve), ("ML Woodbury (lines)", lr_lines)):
        f = ManifoldFitter(noise, pca.basis_, man, lr)
        t = time.time()
        s2, tau2, _ = f.fit(X, s1, tau1)
        timing[name] = time.time() - t
        res[name] = evaluate(s2, E, lab, TEST_LINES, tau2, t0_true)
        out[name] = (f, s2)

    # Cramer-Rao bounds of the learned models at the test lines
    f_lines, s_lines = out["ML Woodbury (lines)"]
    crb = []
    for row_s, row_w, e in zip(res["ML stationary"], res["ML Woodbury (lines)"], TEST_LINES):
        sc = np.median(s_lines[E == e])
        sig_s0, sig_t0 = fit0.crb(np.median(s1[E == e]))
        sig_s, sig_t = f_lines.crb(sc, include_covariance=False)
        sig_sc, sig_tc = f_lines.crb(sc, include_covariance=True)
        crb.append(dict(E=e, stationary=FWHM * sig_s0 / abs(row_s["gain"]),
                        woodbury=FWHM * sig_s / abs(row_w["gain"]),
                        woodbury_cov=FWHM * sig_sc / abs(row_w["gain"]),
                        t_stationary=sig_t0, t_woodbury=sig_t))
    # oracle: per-line Gaussian model estimated on half of the test line (cross-fitted)
    oracle = []
    n = noise.n_samples
    for e in TEST_LINES:
        m = np.where(E == e)[0]
        Xa = aligned(noise, X[m], tau1[m])
        sc = np.median(s1[m])
        c = man.curve([sc])[0] @ pca.basis_
        cp = man.curve([sc], 1)[0] @ pca.basis_
        from empca.shifts import derivative
        Dm = np.vstack([cp, -derivative(c)])
        r = Xa - c
        est = np.empty(len(m))
        halves = (np.arange(len(m)) % 2 == 0, np.arange(len(m)) % 2 == 1)
        for tr_, te_ in (halves, halves[::-1]):
            rt = r[tr_] - r[tr_].mean(0)
            # same model class as the estimator: signal subspace + K_n residual modes
            Pb = np.eye(n) - pca.basis_.T @ pca.basis_
            w, U = np.linalg.eigh(Pb @ (rt.T @ rt / len(rt)) @ Pb)
            V = np.vstack([pca.basis_, U[:, np.argsort(w)[::-1][:Kn]].T])
            Z = rt @ V.T
            S = Z.T @ Z / len(Z) - np.eye(len(V))
            Rinv = np.eye(n) - V.T @ (S @ np.linalg.inv(np.eye(len(V)) + S)) @ V
            F = Dm @ Rinv @ Dm.T
            est[te_] = np.linalg.solve(F, Dm @ Rinv @ r[te_].T)[0]
        g = res["ML stationary"][TEST_LINES.index(e)]["gain"]
        from empca.calibration import line_resolution
        _, fw, err = line_resolution(est)
        oracle.append(dict(E=e, fwhm=fw / abs(g), fwhm_err=err / abs(g)))
    res["Oracle local GLS"] = oracle

    # prior-art baselines (trained on the labelled training lines)
    t = time.time()
    base = PracticeBaselines(of, noise, TRAIN_LINES).fit(D["train"], D["train_lab"])
    timing["learn_baselines"] = time.time() - t
    t = time.time()
    bout, btau = base.transform(X)
    timing["baselines (all four)"] = time.time() - t
    for key, name in (("phase", "OF + phase"), ("decor", "OF + phase + projectors"),
                      ("secant", "Secant filter"), ("covar", "Covariance filter")):
        res[name] = evaluate(bout[key], E, lab, TEST_LINES, btau, t0_true)

    n_test = len(X)
    rates = {k: n_test / v for k, v in timing.items() if not k.startswith("learn")}
    summary = dict(results=res, crb=crb, timing=timing, rates=rates, K=K, Kn=Kn,
                   lr_eigs=lr_lines.pooled_eigenvalues_.tolist(),
                   lambda_manifold=man.lambda_, sipca_iters=len(pca.objective_))
    save_json(f"benchmark_{scenario}.json", summary)
    return summary


def ablations(scenario="morph"):
    D = simulate(scenario)
    noise = StationaryNoise.from_records(D["noise"])
    E, lab, X = D["test_E"], D["test_lab"], D["test"]
    t0_true = D["test_t0"] - 256
    rows = []
    for K in (2, 3, 4, 6, 8):
        pca, man, fit0, s_tr, t0_tr = stage_ab(D, noise, K)
        tau_a, a = pca.transform(X)
        s1, tau1, _ = fit0.fit(X, man.project(a), tau_a)
        for Kn in ((1, 2, 4, 8, 16, 32) if K == 4 else (8,)):
            lr_lines, _, _ = covariances(D, noise, pca, man, s_tr, t0_tr, Kn)
            s2, tau2, _ = ManifoldFitter(noise, pca.basis_, man, lr_lines).fit(X, s1, tau1)
            rows.append(dict(K=K, Kn=Kn,
                             stationary=evaluate(s1, E, lab, TEST_LINES, tau1, t0_true),
                             woodbury=evaluate(s2, E, lab, TEST_LINES, tau2, t0_true)))
            print("ablation", K, Kn, [round(r["fwhm"], 2) for r in rows[-1]["woodbury"]], flush=True)
    save_json(f"ablation_{scenario}.json", rows)


# ------------------------------------------------------------------- figures
LABELS = {"OF": "Optimal filter (5.9 keV template)", "PCA norm": "PCA coefficient norm",
          "OF + phase": "OF + phase correction",
          "OF + phase + projectors": "OF + phase + residual-projector decorrelation",
          "Secant filter": "Secant (tangent) filter, stationary",
          "Covariance filter": "Energy-dependent covariance filter",
          "Manifold projection": "Manifold projection (Stage B)",
          "ML stationary": "Manifold ML, stationary noise",
          "ML Woodbury (curve)": "Manifold ML, Woodbury (label-free)",
          "ML Woodbury (lines)": "Manifold ML, Woodbury (line-aware)",
          "Oracle local GLS": "Oracle local Gaussian model"}
KEYS = {"OF": "of", "PCA norm": "norm", "Manifold projection": "proj", "ML stationary": "stat",
        "ML Woodbury (curve)": "curve", "ML Woodbury (lines)": "lines", "Oracle local GLS": "oracle",
        "OF + phase": "phase", "OF + phase + projectors": "decor", "Secant filter": "secant",
        "Covariance filter": "covar"}
TABLE_ORDER = ("OF", "OF + phase", "OF + phase + projectors", "Secant filter", "Covariance filter",
               "PCA norm", "Manifold projection", "ML stationary", "ML Woodbury (curve)",
               "ML Woodbury (lines)", "Oracle local GLS")
FIG_ORDER = ("OF", "OF + phase", "OF + phase + projectors", "Covariance filter", "ML stationary",
             "ML Woodbury (lines)", "Oracle local GLS")
DASHED = ("OF + phase", "OF + phase + projectors", "Covariance filter", "Oracle local GLS")


def figures():
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=False)
    for ax, sc, title in zip(axes, ("morph", "nomorph"), ("(a) with rise-time variation",
                                                         "(b) without rise-time variation")):
        S = load_json(f"benchmark_{sc}.json")
        for name in FIG_ORDER:
            r = S["results"][name]
            e = np.array([x["E"] for x in r]) / 1e3
            ax.errorbar(e, [x["fwhm"] for x in r], [x["fwhm_err"] for x in r], marker="o", ms=3,
                        color=COLORS[KEYS[name]], label=LABELS[name],
                        ls="--" if name in DASHED else "-", capsize=0)
        ax.set_xlabel("Energy [keV]")
        ax.set_ylabel("FWHM [eV]")
        ax.set_yscale("log")
        ticks = [1.5, 2, 3, 4, 6, 8, 12] if sc == "morph" else [1.5, 2, 2.5, 3, 3.5, 4]
        ax.set_yticks(ticks)
        ax.set_yticks([], minor=True)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.set_title(title, fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(f"{FIG}/resolution.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, sc, title in zip(axes, ("morph", "nomorph"), ("(a) with rise-time variation",
                                                         "(b) without rise-time variation")):
        S = load_json(f"benchmark_{sc}.json")
        for name in ("OF", "ML stationary", "ML Woodbury (lines)"):
            r = S["results"][name]
            ax.plot(np.array([x["E"] for x in r]) / 1e3, np.array([x["jitter"] for x in r]) * 5e3,
                    marker="o", ms=3, color=COLORS[KEYS[name]], label=LABELS[name])
        try:
            P = load_json("timing_phase.json")[sc]
            ax.plot(np.array([x["E"] for x in P]) / 1e3,
                    np.array([x["jitter_phase_removed"] for x in P]) * 5e3, marker="s", ms=2.5,
                    ls="--", color=COLORS["oracle"],
                    label="Woodbury (line-aware), true-phase bias removed (oracle)")
        except FileNotFoundError:
            pass
        e = np.array([x["E"] for x in S["crb"]]) / 1e3
        ax.plot(e, np.array([x["t_stationary"] for x in S["crb"]]) * 5e3, color=COLORS["crb"],
                ls=":", label="CRB, stationary model")
        ax.plot(e, np.array([x["t_woodbury"] for x in S["crb"]]) * 5e3, color=COLORS["crb"],
                ls="-.", label="CRB, learned non-stationary model")
        ax.set_xlabel("Energy [keV]")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel(r"Arrival-time jitter $\sigma_{t_0}$ [ns]")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{FIG}/timing.pdf")

    try:
        rows = load_json("ablation_morph.json")
    except FileNotFoundError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    sel = [r for r in rows if r["Kn"] == 8]
    for key, col, lab in (("stationary", COLORS["stat"], "stationary"),
                          ("woodbury", COLORS["lines"], r"Woodbury (line-aware), $K_n=8$")):
        axes[0].plot([r["K"] for r in sel], [np.mean([x["fwhm"] for x in r[key]]) for r in sel],
                     marker="o", ms=3, color=col, label=lab)
    axes[0].set_xlabel("Subspace dimension $K$")
    axes[0].set_ylabel("Mean FWHM over test lines [eV]")
    axes[0].legend(frameon=False)
    sel = [r for r in rows if r["K"] == 4]
    axes[1].plot([r["Kn"] for r in sel], [np.mean([x["fwhm"] for x in r["woodbury"]]) for r in sel],
                 marker="o", ms=3, color=COLORS["lines"])
    axes[1].axhline(np.mean([x["fwhm"] for x in sel[0]["stationary"]]), color=COLORS["stat"],
                    ls=":", label="stationary")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Number of non-stationary modes $K_n$ ($K=4$)")
    axes[1].set_ylabel("Mean FWHM over test lines [eV]")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/ablation.pdf")


def table():
    """LaTeX table of FWHM per test line for both scenarios."""
    lines = []
    for sc, cap in (("morph", "with rise-time variation"), ("nomorph", "without rise-time variation")):
        S = load_json(f"benchmark_{sc}.json")
        lines.append(r"\multicolumn{%d}{l}{\emph{Scenario %s}}\\" % (len(TEST_LINES) + 1, cap))
        for name in TABLE_ORDER:
            r = S["results"][name]
            lines.append(LABELS[name] + " & " + " & ".join(f"{x['fwhm']:.2f}" for x in r) + r" \\")
        lines.append("CRB, learned model & " + " & ".join(f"{x['woodbury']:.2f}" for x in S["crb"])
                     + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    head = " & ".join(f"{e / 1e3:.1f}" for e in TEST_LINES)
    tex = (r"\begin{tabular}{l" + "r" * len(TEST_LINES) + "}\n\\toprule\n"
           r"Estimator \hfill $E$ [keV] & " + head + r" \\" + "\n\\midrule\n" + "\n".join(lines)
           + "\n\\bottomrule\n\\end{tabular}\n")
    with open(f"{FIG}/../results/table_fwhm.tex", "w") as f:
        f.write(tex)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["run", "ablation", "figures", "table"]
    if "run" in what:
        for sc in ("morph", "nomorph"):
            S = run_scenario(sc)
            for k, v in S["results"].items():
                print(sc, f"{k:24s}", " ".join(f"{x['fwhm']:6.2f}" for x in v), flush=True)
            print(sc, "rates [events/s]", {k: round(v) for k, v in S["rates"].items()})
    if "ablation" in what:
        ablations()
    if "figures" in what:
        figures()
    if "table" in what:
        table()
