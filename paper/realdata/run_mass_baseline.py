"""External baseline: NIST MASS's own standard pipeline on the real data.

Run in an environment with MASS installed
(``pip install git+https://github.com/usnistgov/mass``).  Pipeline, as in
MASS's own tests: summarize -> auto_cuts -> compute_noise ->
avg_pulses_auto_masks -> filter (ATS with f_3db = 10 kHz, and 5-lag) ->
filter_data -> drift_correct -> phase_correct.

MASS fits its corrections on *all* records (it has no train/test split), so its
widths below are in-sample and, if anything, optimistic.  Widths are measured
on the same held-out half (odd record indices) and line window as the other
estimators, as a fraction of the line centroid.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from empca.calibration import line_resolution  # noqa: E402

import mass  # noqa: E402

DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")


def run(filter_kind):
    tmp = tempfile.mkdtemp()
    shutil.copy(os.path.join(DATA, "phase_correct_test_data_4k_pulses_chan1.ljh"),
                os.path.join(tmp, "real_chan1.ljh"))
    shutil.copy(os.path.join(DATA, "regress_noise_chan1.ljh"), os.path.join(tmp, "real_noise_chan1.ljh"))
    data = mass.TESGroup(os.path.join(tmp, "real_chan1.ljh"), os.path.join(tmp, "real_noise_chan1.ljh"))
    data.summarize_data(forceNew=True)
    data.auto_cuts(forceNew=True, clearCuts=True)
    data.compute_noise()
    data.avg_pulses_auto_masks()
    if filter_kind == "ats":
        data.compute_ats_filter(f_3db=10000)
    else:
        data.compute_5lag_filter()
    data.filter_data()
    data.drift_correct(forceNew=True)
    data.phase_correct(forceNew=True)
    ds = data.channel[1]
    out = {}
    n = ds.nPulses
    te = np.arange(n) % 2 == 1
    good = ds.good()
    for name, attr in (("filt", "p_filt_value"), ("dc", "p_filt_value_dc"), ("phc", "p_filt_value_phc")):
        v = np.asarray(getattr(ds, attr)[:], float)
        c = np.median(v[good])
        line = np.abs(v / c - 1) < 0.02
        for cut, m in (("all", te & line), ("mass_cuts", te & line & good)):
            mu, fw, err = line_resolution(v[m])
            out[f"{name}_{cut}"] = dict(fwhm=100 * fw / mu, err=100 * err / mu, n=int(m.sum()))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    res = {"mass_version": mass.__version__}
    for kind in ("ats", "5lag"):
        res[kind] = run(kind)
        print(kind, {k: round(v["fwhm"], 4) for k, v in res[kind].items()})
    with open(os.path.join(RESULTS, "realdata_mass.json"), "w") as f:
        json.dump(res, f, indent=1)
