# Prior-art review (October 2026)

## How this review was done

- **Literature:** a web search of journals, arXiv, NASA ADS, NIST and NTRS records.
- **Software:** GitHub and PyPI searches, plus shallow clones of NIST `mass`/`mass2`,
  SuperCDMS `QETpy`/`detprocess` and CRESST `cait`, whose sources were grepped.
- **Limitation:** full texts on journal sites and arXiv could not be opened from the build
  machine, so several judgements rest on abstracts and search snippets. These are
  marked **(snippet)**. The items under "To verify before submission" need a full read.

**Verdict.** No paper or public code combines (A) shift-invariant whitened PCA with
alternating sub-sample alignment, (B) a spline pulse manifold in whitened arc length and
(C) a learned low-rank, energy-dependent covariance inside a Woodbury-accelerated joint
(E, t0) maximum-likelihood fit. Each stage has close antecedents, which the paper cites
and positions against (Sec. 1 and Sec. 9).

## A. Shift-invariant whitened PCA

| Work | What it does | Overlap |
|---|---|---|
| Yu 2019, arXiv:1910.14261 (this author) | EM-PCA in a phase-differenced Fourier domain; time shift linearised, not estimated | Partial; the project's own predecessor |
| NIST `mass`/`mass2` (MIT): `ToeplitzWhitener`, `pulse_model.py` projectors, `Filter5Lag`, `PhaseCorrector` | Whitened OF; SVD "projector" basis of residuals, documented as *non-noise-weighted*; arrival time as a derivative column or 5-lag parabola; empirical phase correction | Partial; no alternating re-alignment and re-SVD |
| Fowler et al., JLTP 199, 745 (2020) | Robust PCA with Cholesky whitening; a component absorbs early/late arrival linearly (snippet) | Partial |
| Alpert et al., JLTP 184, 263 (2016), arXiv:1512.01608 | SVD pulse model with amplitude, sub-sample arrival and baseline, for pile-up (HOLMES) | Partial |
| *Nuclear Techniques* 49 (2026), doi:10.3724/j.0253-3219.2026.hjs.49.250425 | Cross-correlation alignment, then a single PCA (snippet) | Partial; not iterative |
| Frey & Jojic (TPAMI ~2003); Kannan, Jojic & Frey (NIPS 2002) | Transformation-invariant clustering / factor analysis by EM with FFT shifts | **Same algorithmic core**, outside detectors; no PSD whitening or sub-sample phase |
| Mørup et al., NeuroImage (2008); Kneip & Ramsay, JASA 103 (2008) | Shift-invariant decompositions with frequency-domain delays; registration plus PCA | Substantial (mechanism) |

## B. Pulse manifold

| Work | What it does | Overlap |
|---|---|---|
| Fixsen et al., JLTP 176, 16 (2014) | Differential-geometric, energy-dependent least squares on the pulse curve | Substantial (concept) |
| Fowler et al., IEEE TAS 27 (2017), arXiv:1611.07856 | "Tangent filtering" along the energy curve of mean pulses | Substantial (concept); not learned or shift-invariant |
| Fowler et al., JLTP 193, 539 (2018), arXiv:1803.02827 | Cubic splines of linear-subspace components versus an energy estimator | Close; no public code found |
| Yan et al., JLTP 184, 397 (2016); Miller et al., JATIS 7, 048003 (2021) | Energy axis from PCA coefficients and calibration lines (linear; N components) | Partial |
| SIRENA (SIXTE/HEASoft) `PCA` method | Linear PC1–PC2 axis between two energies; documented as not fully implemented | Partial |
| Ichinohe et al., JLTP 209 (2022) | VAE latent space captures non-linearity; used for pulse tagging | Weak |

What appears new: the curve is a P-spline graph through the origin in whitened
coefficient space; its parameter is whitened arc length (equal to the matched-filter
SNR, with unit Fisher information); and the low-energy estimate is unbiased by
construction.

## C. Non-stationary maximum likelihood

| Work | What it does | Overlap |
|---|---|---|
| Fixsen, Moseley, Cabrera, Figueroa-Feliciano, AIP CP 605, 339 (2002); NIM A 520, 555 (2004) | Energy-dependent dense covariance in a least-squares fit | Substantial (R(E)); dense O(N^3) |
| Shank et al., AIP Adv. 4, 117106 (2014) | Monte-Carlo energy-dependent covariance, χ²(E) scan | Substantial; covariance simulated |
| Peille et al., SPIE 9905, 99055W (2016); SIRENA `WEIGHT/WEIGHTN` (`INTCOVAR/COVAR`) | Full covariances at calibration lines, interpolated; best resolution but impractical calibration burden | Substantial; full-rank, no joint t0 |
| SuperCDMS non-stationary OF (CDMSlite, arXiv:1707.01632) | Template-residual covariance aligned to t0, added to the noise | Substantial (closest in spirit); dense |
| Albakry et al. (SuperCDMS), NIM A (2026), arXiv:2508.20090 | N×M filter with PCA templates; shape uncertainty treated as non-stationary noise | Partial to substantial; modes profiled, no Woodbury |
| van Haasteren & Vallisneri, PRD 90, 104012 (2014); Lentati et al., MNRAS 466, 3706 (2017) | Pulsar timing: white noise plus low-rank GP via Woodbury and the determinant lemma; joint TOA and shape variation | **Same mathematics**, other field |
| Moore & Gair, PRL 113, 251101 (2014) | Waveform error marginalised as GP covariance (gravitational waves) | Analogue |

What appears new: a learned, low-rank, continuous Σ(E) that includes the
signal subspace; a joint (E, t0) likelihood reduced to K_b dimensions (with data
efficiency shown against full-rank covariances); and the identifiability statement
for tangential fluctuations.

## Software surveyed

| Codebase | Relevant content |
|---|---|
| `usnistgov/mass`, `mass2`, `dastard` (NIST) | Practitioner reference: whitened OF, projectors, phase and drift correction |
| `spice-herald/QETpy`, `detprocess` (SuperCDMS) | OF variants including `OFnonlin`, N×M and pile-up; PSD-only TES noise model; no PCA or manifold |
| SIRENA, `tessim`, `xifusim` (Athena X-IFU, SIXTE) | Covariance-matrix and PCA energy methods; physical TES simulator |
| `fewagner/cait` (CRESST) | Unweighted PCA features; template fits |
| `MazinLab/VENOM` | Neural (Mamba) energy estimator for MKIDs |
| `sbailey/empca` | Generic weighted EM-PCA |
| `relmt` (seismology) | Iterative SVD and sub-sample waveform alignment |

No public code from ECHo, HOLMES, CUORE/CUPID or Ricochet was found.

## To verify before submission (full text needed)

1. Fixsen 2014: whether the covariances were estimated at photon-number peaks, and how
   arc length is used.
2. SuperCDMS non-stationary OF and Albakry 2026: confirm there is no low-rank or Woodbury form.
3. Miller 2021: the exact coefficient-to-energy mapping.
4. *Nuclear Techniques* 2026: authors, volume and method details.
5. Bibliographic details marked uncertain in `refs.bib` (Fowler 2017 article number;
   Busch 2016 author order and pages).
