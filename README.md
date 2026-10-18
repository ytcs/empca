# empca — template-free optimal pulse processing for microcalorimeters

`empca` estimates the energy and arrival time of microcalorimeter (TES, MMC, MKID)
pulses **without a calibrated template**. It stays optimal when pulses saturate
and change shape, when the noise is non-stationary, and when arrival times jitter
at the sub-sample level.

It is the follow-up to
[*Template-free Pulse Height Estimation of Microcalorimeter Responses with PCA*](https://arxiv.org/abs/1910.14261)
(T. C. Yu, 2019). The original frequency-domain EM-PCA is still available as
`empca.EMPCA`.

| Stage | Class | What it does |
|---|---|---|
| — | `StationaryNoise` | noise spectrum and shift-commuting spectral whitening `R0^{-1/2}` |
| A | `ShiftInvariantPCA` | whitened shift-invariant EM-PCA: phase-coherent cross-correlation E-step (sub-sample Newton), counter-rotated SVD M-step |
| B | `PulseManifold` | P-spline curve through the origin in the coefficient space, parameterised by whitened arc length (= cumulative matched-filter SNR, unit Fisher information) |
| C | `LowRankNoise`, `ManifoldFitter` | energy-dependent low-rank covariance `R(E,t0) = R0 + T_t0 G Σ(E) Gᵀ T_t0ᵀ`; joint ML of (E, t0) with Woodbury / determinant-lemma likelihood in `K_b ≈ 12` dimensions, plus Cramér–Rao bounds |
| — | `PulseProcessor` | scikit-learn-style pipeline chaining all of the above |
| — | `OptimalFilter` | classical OF with arrival-time search (baseline) |
| — | `empca.simulate.TESSimulator` | Euler–Maruyama electro-thermal TES simulator (saturation, non-stationary Johnson/phonon noise, position-dependent rise, continuous arrival times) |

## Install

```bash
pip install -e .            # numpy, scipy
pip install -e ".[test]"    # + pytest
```

## Usage

```python
from empca import PulseProcessor

proc = PulseProcessor(n_components=4, n_lines=7, n_noise_modes=8)
proc.fit(train_records, noise_records)           # unlabelled pulses + pulse-free records
res = proc.transform(records)                    # res.s, res.t0, res.nll
proc.calibrate(proc.line_centers_, line_energies)  # map manifold coordinate -> eV
energy = proc.energy(res.s)
fwhm = proc.resolution(res.s)                    # Cramér–Rao FWHM of the learned model
template = proc.template(res.s[0])               # noise-free pulse at that energy
```

`examples/quickstart.py` simulates a saturating TES and compares the result
with the optimal filter.

### Choosing the covariance model

* `covariance="lines"` (default) learns the non-stationary noise from the
  mono-energetic clusters it finds in the data. Their energies need not be
  known. This is the only variant that captures shape fluctuations *along* the
  manifold, which broaden lines.
* `covariance="curve"` is fully label-free. It uses residuals about the manifold
  and so only sees fluctuations orthogonal to it. These matter for weighting and
  timing, but to first order they do not change the energy estimate (see the
  identifiability proposition in the paper).
* `covariance=None` gives the stationary-noise ML fit.

## Results on a simulated saturating TES (held-out test lines)

FWHM in eV, with ±2 % absorber thermalisation spread. The test lines lie between
the training lines.

| FWHM [eV] at | 1.0 keV | 3.7 keV | 6.5 keV | 9.0 keV | 11.0 keV |
|---|---|---|---|---|---|
| Optimal filter (5.9 keV template) | 2.01 | 4.34 | 7.24 | 9.99 | 12.40 |
| OF + phase correction + residual-projector decorrelation | 1.73 | 1.91 | 2.12 | 2.56 | 3.10 |
| Energy-dependent full-covariance filter (Fixsen / SIRENA style) | 1.76 | 2.19 | 2.72 | 3.00 | 3.30 |
| **this work** (line-aware Woodbury ML) | **1.71** | **1.91** | **2.04** | **2.30** | **2.56** |
| oracle local Gaussian model | 1.74 | 1.92 | 2.05 | 2.31 | 2.57 |

**Against a plain OF** the gain is large, but most of it is also available from a
well-tuned empirical correction chain.

**Against the best prior-art baseline** at each line the method is level at low
energy and up to 18 % better at high energy. With no shape fluctuation at all it is
up to 8 % better.

**Against a full-covariance filter** trained on the same calibration pulses, the
low-rank model needs far fewer pulses.

On **real NIST TES data** distributed with MASS (one dominant line, held-out half),
`paper/realdata/run_realdata.py` measures FWHM as a fraction of the line energy:

| Estimator | FWHM |
|---|---|
| OF | 0.46 % |
| NIST MASS standard pipeline (ATS filter + drift + phase; in-sample) | **0.254 %** |
| NIST MASS (5-lag filter + drift + phase; in-sample) | 0.265 % |
| this work (line-aware, + pre-trigger drift covariate) | 0.262 % |
| baseline-noise limit | 0.217 % |

On real single-line data the method **matches but does not beat** NIST's own
pipeline. MASS beats the author's reimplementation of the same corrections, so the
simulated baselines may understate real practice.

`PhaseEqualizer` removes the sampling-phase bias of arrival times without labels,
given about 10^5 records per energy.

**Phonon-mediated detector** (simulated, strong position dependence, line-free science
data; `paper/scripts/run_phonon.py`): with the covariance learned on a calibration
run, the method is 1.2–3.6× better than an energy-dependent full-covariance filter
within the calibrated range (e.g. 33 vs 52 eV at 5 keV). A model trained on the
science data alone cannot remove the position dependence, as the identifiability
result predicts.

See `paper/` for the full study and `paper/PRIOR_ART.md` for the prior-art review.

## Paper

`paper/` contains the LaTeX manuscript (`main.tex`). `paper/scripts/` regenerates
every figure, table and quoted number:

```bash
cd paper
make data      # simulations + analyses (~30 min, 4 cores)
make figures   # plots, table and numbers.tex from saved results
make pdf
```

## Tests

```bash
pytest
```

Please cite the paper if you use this library in research.
