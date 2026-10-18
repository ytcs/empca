"""Real-data test on NIST TES data distributed with MASS (MIT licence).

Data: APS beamline 29, 13 Aug 2015, NIST TES "chan101" (5.12 us sampling,
1024 samples, 512 pre-trigger), ~4000 pulse records dominated by one line
(file name ``..._600eV_...``) and 300 noise records from the same channel.
MASS publishes this pulse file as its arrival-phase-correction test data.

Protocol: alternate records form training and test halves.  Everything is
learned on the training half; widths are measured on the test half.  With
essentially one line the absolute energy scale is unknown, so each
estimator's line width is converted to a fraction of the line energy with its
local gain against the OF amplitude, measured on the off-line events between
0.85 and 1.02 of the line (the real-data analogue of the simulation's
continuum gain).
"""
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from common import DATA, FIG, RESULTS, COLORS, mpl  # noqa: E402
from baselines import PracticeBaselines  # noqa: E402
from ljh import read_ljh  # noqa: E402
from empca import OptimalFilter, PulseProcessor, StationaryNoise, line_resolution  # noqa: E402
from empca.calibration import FWHM  # noqa: E402

BASE = "https://raw.githubusercontent.com/usnistgov/mass/master/tests/regression_test/"
FILES = {"pulses": "phase_correct_test_data_4k_pulses_chan1.ljh", "noise": "regress_noise_chan1.ljh"}


def fetch():
    paths = {}
    for k, f in FILES.items():
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(BASE + f, p)
        paths[k] = p
    return paths


def main():
    paths = fetch()
    X, info = read_ljh(paths["pulses"])
    Nz, _ = read_ljh(paths["noise"])
    X = X - X[:, :450].mean(1, keepdims=True)
    Nz = Nz - Nz.mean(1, keepdims=True)
    Nz = Nz[(Nz.max(1) - np.median(Nz, 1)) < 150]           # drop noise records with pulses
    noise = StationaryNoise.from_records(Nz)
    tr = np.arange(len(X)) % 2 == 0
    te = ~tr

    ph = X.max(1)
    of = OptimalFilter(max_shift=10).fit(X[tr & (ph > 13380) & (ph < 13520)], noise)
    A, tau = of.transform(X)
    line = (A > 0.98) & (A < 1.02)
    tail = (A > 0.85) & (A < 1.02)

    def width(v, m=te & line, gain=None):
        g = np.polyfit(A[tail], v[tail], 1)[0] if gain is None else gain
        mu, fw, err = line_resolution(v[m])
        return dict(fwhm=100 * fw / g, err=100 * err / g)

    res = {"OF": width(A)}
    lab = np.where(line, 0, -1)
    base = PracticeBaselines(of, noise, [1.0]).fit(X[tr], lab[tr])
    bout, _ = base.transform(X)
    # corrections fitted at the line keep the OF gain there (their off-line
    # extrapolation is not defined with a single line), so no tail regression
    res["OF + phase"] = width(bout["phase"], gain=1.0)
    res["OF + phase + projectors"] = width(bout["decor"], gain=1.0)
    for cov, name in ((None, "ML stationary"), ("lines", "ML Woodbury (lines)")):
        proc = PulseProcessor(n_components=4, max_shift=10, snr_min=30, n_lines=1,
                              n_noise_modes=8, covariance=cov).fit(X[tr], noise=noise)
        r = proc.transform(X)
        res[name] = width(r.s)
        if cov == "lines":
            jit_of = np.std(tau[te & line] - np.median(tau[te & line]))
            res["manifold_mode"] = proc.manifold_.mode_
    res["baseline_limit"] = dict(fwhm=100 * FWHM * of.resolution, err=0.0)
    res["n_train_line"] = int((tr & line).sum())
    res["n_test_line"] = int((te & line).sum())
    res["n_noise"] = int(len(Nz))
    with open(os.path.join(RESULTS, "realdata.json"), "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, v)

    plt = mpl()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    names = ["OF", "OF + phase", "OF + phase + projectors", "ML stationary", "ML Woodbury (lines)"]
    labels = ["OF", "OF + phase", "OF + phase\n+ projectors", "Manifold ML\n(stationary)",
              "Manifold ML\n(line-aware)"]
    cols = [COLORS[k] for k in ("of", "phase", "decor", "stat", "lines")]
    vals = [res[n]["fwhm"] for n in names]
    errs = [res[n]["err"] for n in names]
    ax.bar(range(len(vals)), vals, yerr=errs, color=cols, width=0.65)
    ax.axhline(res["baseline_limit"]["fwhm"], color=COLORS["crb"], ls=":", label="baseline-noise limit")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("FWHM [% of line energy]")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "realdata.pdf"))


if __name__ == "__main__":
    main()
