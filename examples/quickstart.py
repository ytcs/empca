"""Quick start: simulate a saturating TES, learn the model without labels,
calibrate with known lines and compare with the optimal filter.

    python examples/quickstart.py        (about 2 minutes)
"""
import numpy as np

from empca import (EnergyCalibration, OptimalFilter, PulseProcessor, find_clusters,
                   line_resolution)
from empca.simulate import TESSimulator

LINES = [1500., 3000., 4500., 5900., 8000., 10000., 12000.]

sim = TESSimulator(seed=1)                       # physical TES model, see empca.simulate
train = sim.spectrum(8000, lines=LINES)          # unlabelled training spectrum
test = sim.spectrum(8000, lines=LINES)           # independent test spectrum
noise = sim.noise(1500)                          # pulse-free records

# --- learn: shift-invariant whitened PCA -> manifold -> line-aware Woodbury ML
proc = PulseProcessor(n_components=4, max_shift=8, snr_min=30,
                      n_lines=len(LINES), n_noise_modes=8)
proc.fit(train.traces, noise)
proc.calibrate(proc.line_centers_, LINES)        # only step that needs energies

res = proc.transform(test.traces)
energy = proc.energy(res.s)

# --- reference: optimal filter with a 5.9 keV template and the same calibration
of = OptimalFilter(max_shift=8).fit(train.traces[train.energy == 5900.], proc.noise_)
amp_train, _ = of.transform(train.traces)
_, centers = find_clusters(amp_train, len(LINES))
amp, _ = of.transform(test.traces)
energy_of = EnergyCalibration(centers, LINES)(amp)

print(f"{'line [eV]':>10} {'OF FWHM':>9} {'this work':>10}  [eV]")
for e in LINES:
    m = test.energy == e
    print(f"{e:10.0f} {line_resolution(energy_of[m])[1]:9.2f} {line_resolution(energy[m])[1]:10.2f}")
