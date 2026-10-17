"""Shared configuration, cached simulations and evaluation helpers for the paper."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))

from empca.calibration import line_resolution  # noqa: E402
from empca.simulate import TESParameters, TESSimulator  # noqa: E402

DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")
for d in (DATA, FIG, RESULTS):
    os.makedirs(d, exist_ok=True)

TRAIN_LINES = (1500., 3000., 4500., 5900., 8000., 10000., 12000.)
TEST_LINES = (1000., 2200., 3700., 5000., 6500., 7200., 9000., 11000.)
N_TRAIN, N_TEST, N_NOISE = 20000, 20000, 3000

SCENARIOS = {
    # position-dependent thermalisation (rise-time) spread of +-2 %
    "morph": dict(eta_scale=0.02),
    # no morphology variation: nonlinearity + physical non-stationary noise only
    "nomorph": dict(eta_scale=0.0),
}


def simulate(scenario, seed=0):
    """Training spectrum, independent test spectrum and noise records (cached)."""
    path = os.path.join(DATA, f"sim_{scenario}_{seed}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    sim = TESSimulator(TESParameters(**SCENARIOS[scenario]), seed=seed)
    tr = sim.spectrum(N_TRAIN, lines=TRAIN_LINES)
    te = sim.spectrum(N_TEST, lines=TEST_LINES)
    nz = sim.noise(N_NOISE)
    out = dict(train=tr.traces, train_E=tr.energy, train_eta=tr.eta, train_t0=tr.t0,
               train_lab=tr.labels, test=te.traces, test_E=te.energy, test_eta=te.eta,
               test_t0=te.t0, test_lab=te.labels, noise=nz)
    np.savez(path, **out)
    return out


def local_gain(x, E, lab, e, half=600.0):
    """d<x>/dE at energy e from continuum events (quadratic local regression)."""
    m = (lab < 0) & (np.abs(E - e) < half)
    A = np.vstack([np.ones(m.sum()), E[m] - e, (E[m] - e) ** 2]).T
    return np.linalg.lstsq(A, x[m], rcond=None)[0][1]


def evaluate(x, E, lab, lines, tau=None, t0=None):
    """FWHM [eV] (with standard error) and timing jitter per line."""
    out = []
    for e in lines:
        m = E == e
        g = local_gain(x, E, lab, e)
        _, fw, err = line_resolution(x[m])
        row = dict(E=e, fwhm=fw / abs(g), fwhm_err=err / abs(g), gain=g)
        if tau is not None:
            d = tau[m] - t0[m]
            d = d - np.median(d)
            row["jitter"] = float(1.4826 * np.median(np.abs(d)))
        out.append(row)
    return out


def save_json(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "font.family": "serif", "mathtext.fontset": "cm",
    })
    return plt


# colour-blind-safe categorical palette (Okabe-Ito)
COLORS = dict(of="#000000", norm="#999999", proj="#56B4E9", stat="#0072B2",
              curve="#E69F00", lines="#D55E00", crb="#009E73", oracle="#CC79A7",
              phase="#999999", decor="#56B4E9", secant="#0072B2", covar="#E69F00")


def trained(scenario="morph", K=4, Kn=8):
    """Stage A/B/C models learned on the training spectrum (cached with pickle)."""
    import pickle
    from empca import (LowRankNoise, ManifoldFitter, PulseManifold, ShiftInvariantPCA,
                       StationaryNoise, find_clusters)
    from empca.shifts import shift_rfft
    path = os.path.join(DATA, f"models_{scenario}_K{K}_Kn{Kn}_inc.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    D = simulate(scenario)
    noise = StationaryNoise.from_records(D["noise"])
    pca = ShiftInvariantPCA(K, n_align=1, max_shift=8, snr_min=30.0).fit(D["train"], noise)
    man = PulseManifold().fit(pca.coeffs_)
    fit0 = ManifoldFitter(noise, pca.basis_, man)
    s, t0, _ = fit0.fit(D["train"], man.s_, pca.t0_)
    n = noise.n_samples
    Xa = np.fft.irfft(shift_rfft(noise.whiten_rfft(D["train"]), -t0, n), n=n, axis=1)
    groups, _ = find_clusters(s, len(TRAIN_LINES))
    lr_lines = LowRankNoise(Kn).fit(Xa, s, groups=groups, include=pca.basis_)
    lr_curve = LowRankNoise(Kn).fit(Xa - man.curve(s) @ pca.basis_, s, include=pca.basis_)
    out = dict(noise=noise, pca=pca, man=man, fit0=fit0, s=s, t0=t0, groups=groups,
               lr_lines=lr_lines, lr_curve=lr_curve)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out
