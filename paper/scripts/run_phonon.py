"""Phonon-mediated (dark-matter style) detector: line-free science data.

A calibration run (mono-energetic lines on a Compton-like continuum) and an
independent science run without lines (flat + exponential background) are
simulated with PhononSimulator, which has strong position dependence of both
amplitude and pulse shape.  Every estimator is trained on what a real
experiment would have, and is scored on the *science* run from the simulation
truth: in bins of true energy, the FWHM of the energy error after removing a
local linear trend (so that calibration interpolation bias is not counted).

Estimators
----------
* OF and the four practitioner baselines, trained on the calibration run.
* This work, label-free: trained on the science run only (curve covariance).
* This work, line-aware, trained on the calibration run only (calibration-line
  identities given, as for the baselines).
* This work, line-aware, trained on calibration + science runs together
  (science continuum constrains the curve; calibration lines the covariance).
All energy scales come from the calibration lines.
"""
import os

import numpy as np

from common import COLORS, DATA, FIG, load_json, mpl, save_json
from baselines import PracticeBaselines
from empca import EnergyCalibration, OptimalFilter, PulseProcessor, StationaryNoise
from empca.calibration import FWHM, robust_sigma
from empca.simulate import PhononParameters, PhononSimulator

CAL_LINES = (1300., 5900., 10400., 14400., 17500., 22200.)
N_CAL, N_SCI, N_NOISE = 20000, 20000, 3000
EDGES = np.array([0.5, 1.5, 2.5, 4, 6, 8, 10, 12.5, 15, 17.5, 20, 22.5, 25]) * 1e3


def simulate(seed=5):
    path = os.path.join(DATA, f"phonon_{seed}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    sim = PhononSimulator(PhononParameters(), seed=seed)
    cal = sim.calibration_run(N_CAL, lines=CAL_LINES)
    sci = sim.science_run(N_SCI)
    nz = sim.noise(N_NOISE)
    out = dict(cal=cal.traces, cal_E=cal.energy, cal_lab=cal.labels, cal_pos=cal.eta,
               sci=sci.traces, sci_E=sci.energy, sci_pos=sci.eta, sci_t0=sci.t0, noise=nz)
    np.savez(path, **out)
    return out


def score(E_hat, E_true):
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (E_true >= lo) & (E_true < hi) & np.isfinite(E_hat)
        r = E_hat[m] - E_true[m]
        A = np.vstack([np.ones(m.sum()), E_true[m] - 0.5 * (lo + hi)]).T
        res = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
        _, sd = robust_sigma(res)
        rows.append(dict(E=0.5 * (lo + hi), fwhm=FWHM * sd, err=FWHM * sd / np.sqrt(2 * m.sum()),
                         n=int(m.sum()), bias=float(np.median(r))))
    return rows


def calibrate(x_cal, lab, x_sci):
    centres = [np.median(x_cal[lab == i]) for i in range(len(CAL_LINES))]
    return EnergyCalibration(centres, CAL_LINES)(x_sci)


def run():
    D = simulate()
    noise = StationaryNoise.from_records(D["noise"])
    cal, lab, sci, Et = D["cal"], D["cal_lab"], D["sci"], D["sci_E"]
    res = {}
    of = OptimalFilter(max_shift=10).fit(cal[lab == 1], noise)
    A_cal, _ = of.transform(cal)
    A_sci, _ = of.transform(sci)
    res["OF"] = score(calibrate(A_cal, lab, A_sci), Et)

    base = PracticeBaselines(of, noise, CAL_LINES).fit(cal, lab)
    b_cal, _ = base.transform(cal)
    b_sci, _ = base.transform(sci)
    for key, name in (("phase", "OF + phase"), ("decor", "OF + phase + projectors")):
        res[name] = score(calibrate(b_cal[key], lab, b_sci[key]), Et)
    res["Secant filter"] = score(b_sci["secant"], Et)
    res["Covariance filter"] = score(b_sci["covar"], Et)

    kw = dict(n_components=4, max_shift=10, snr_min=30, n_noise_modes=8)
    # label-free: science data only
    p = PulseProcessor(covariance="curve", **kw).fit(sci, noise=noise)
    res["This work, label-free (science only)"] = score(
        calibrate(p.transform(cal).s, lab, p.transform(sci).s), Et)
    # line-aware, calibration run only; the calibration-line identities are given,
    # exactly as for the practitioner baselines
    p = PulseProcessor(covariance="lines", **kw).fit(cal, noise=noise, groups=lab)
    res["This work, line-aware (calibration run)"] = score(
        calibrate(p.transform(cal).s, lab, p.transform(sci).s), Et)
    # line-aware, calibration + science
    p = PulseProcessor(covariance="lines", **kw).fit(
        np.vstack([cal, sci]), noise=noise, groups=np.r_[lab, -np.ones(len(sci), int)])
    res["This work, line-aware (calibration + science)"] = score(
        calibrate(p.transform(cal).s, lab, p.transform(sci).s), Et)
    save_json("phonon.json", res)
    for k, v in res.items():
        print(f"{k:46s}", " ".join(f"{x['fwhm']:7.1f}" for x in v), flush=True)


def figure():
    res = load_json("phonon.json")
    plt = mpl()
    style = {"OF": ("of", "-"), "OF + phase": ("phase", "--"),
             "OF + phase + projectors": ("decor", "--"), "Covariance filter": ("covar", "--"),
             "This work, label-free (science only)": ("curve", "-"),
             "This work, line-aware (calibration run)": ("stat", "-"),
             "This work, line-aware (calibration + science)": ("lines", "-")}
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for name, (c, ls) in style.items():
        r = res[name]
        ax.errorbar(np.array([x["E"] for x in r]) / 1e3, [x["fwhm"] for x in r],
                    [x["err"] for x in r], color=COLORS[c], ls=ls, marker="o", ms=2.5, label=name)
    ax.set_xlabel("true energy [keV]")
    ax.set_ylabel("FWHM of energy error [eV]")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/phonon.pdf")


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["run", "figure"]
    if "run" in what:
        run()
    if "figure" in what:
        figure()
