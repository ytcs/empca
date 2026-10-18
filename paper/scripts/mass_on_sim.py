"""External baseline on the simulation: NIST MASS's standard pipeline applied to
the simulated *test* spectra (written as LJH files).

STATUS: not used in the paper.  With MASS's default settings this does not
reproduce even the plain optimal-filter resolution on our simulated records
(e.g. 2.3 eV vs 1.7 eV at 1 keV before any correction), which points to a
configuration mismatch rather than a property of MASS.  MASS is therefore
compared only on real data (paper/realdata/run_mass_baseline.py).

Run in an environment with MASS installed.  MASS fits its filter template,
drift and phase corrections on the test spectrum itself (in-sample), which can
only favour it.  Widths are converted to eV with the same local-gain procedure
as every other estimator (common.evaluate).
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "realdata"))
from common import TEST_LINES, evaluate, save_json, simulate  # noqa: E402
from ljh import write_ljh  # noqa: E402

import mass  # noqa: E402

# arrivals are at 256 +- 2 samples; MASS assumes the pulse starts after the
# pre-trigger window, as in real triggered data
PRESAMPLES = 250


def run(scenario, filter_kind):
    D = simulate(scenario)
    tmp = tempfile.mkdtemp()
    write_ljh(os.path.join(tmp, "sim_chan1.ljh"), D["test"], 5e-6, PRESAMPLES)
    write_ljh(os.path.join(tmp, "sim_noise_chan1.ljh"), D["noise"], 5e-6, PRESAMPLES)
    data = mass.TESGroup(os.path.join(tmp, "sim_chan1.ljh"), os.path.join(tmp, "sim_noise_chan1.ljh"))
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
    for name, attr in (("filt", "p_filt_value"), ("phc", "p_filt_value_phc")):
        v = np.asarray(getattr(ds, attr)[:], float)
        t = np.asarray(ds.p_filt_phase[:], float)
        out[name] = evaluate(v, D["test_E"], D["test_lab"], TEST_LINES, t, D["test_t0"] - 256)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    res = {"mass_version": mass.__version__, "presamples": PRESAMPLES}
    scen = sys.argv[1:] or ["morph", "nomorph"]
    for sc in scen:
        for kind in ("ats", "5lag"):
            r = run(sc, kind)
            res[f"{sc}_{kind}"] = r
            for k, v in r.items():
                print(sc, kind, k, " ".join(f"{x['fwhm']:.2f}" for x in v), flush=True)
    save_json("mass_on_sim.json", res)
