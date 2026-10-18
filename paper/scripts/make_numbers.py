"""Write results/numbers.tex: every number quoted in the text, from the result files."""
import os

import numpy as np

from common import RESULTS, TEST_LINES, load_json, trained
from empca.simulate import TESSimulator


def fmt(x, nd=1):
    out = f"{x:.{nd}f}"
    return out[1:] if out.startswith("-") and float(out) == 0 else out


def main():
    m = load_json("benchmark_morph.json")
    n = load_json("benchmark_nomorph.json")
    R, Rn = m["results"], n["results"]
    fw = lambda r, name: np.array([x["fwhm"] for x in r[name]])
    lines, of, stat, orc = (fw(R, k) for k in ("ML Woodbury (lines)", "OF", "ML stationary",
                                                "Oracle local GLS"))
    lines_n, of_n = fw(Rn, "ML Woodbury (lines)"), fw(Rn, "OF")
    i65 = TEST_LINES.index(6500.)
    gain = of / lines
    gain_n = of_n / lines_n
    jit = [x["jitter"] for name in ("OF", "ML stationary", "ML Woodbury (lines)")
           for x in R[name]]
    crb_t_n = np.array([x["t_woodbury"] for x in n["crb"]])
    jit_n = np.array([x["jitter"] for x in Rn["ML Woodbury (lines)"]])
    sim = TESSimulator()
    peaks = {}
    for e in (1000., 12000.):
        x = TESSimulator(seed=11).simulate(np.full(200, e)).mean(axis=0)
        peaks[e] = x.max() / e
    M = trained("morph")
    low = load_json("lowenergy.json")
    est = low["estimators"]
    norm0 = est["PCA coefficient norm"]["mean"][0]
    of0 = est["Optimal filter"]["mean"][0]
    ml0 = est["Manifold ML (this work)"]["mean"][0]
    ml_sd_ratio = est["Manifold ML (this work)"]["std"][0] / est["Optimal filter"]["std"][0]
    P = load_json("timing_phase.json")
    pn = np.array([x["jitter_phase_removed"] for x in P["nomorph"]])
    crb_n = np.array([x["t_woodbury"] for x in n["crb"]])
    jm = np.array([x["jitter"] for x in R["ML Woodbury (lines)"]]) * 5e3
    wb = load_json("woodbury_benchmark.json")
    w1024 = [r for r in wb if r["N"] == 1024][0]
    BASE = ("OF + phase", "OF + phase + projectors", "Secant filter", "Covariance filter")
    best_m = np.min([fw(R, b) for b in BASE], axis=0)
    best_n = np.min([fw(Rn, b) for b in BASE], axis=0)
    imp_m = 100 * (1 - lines / best_m)
    imp_n = 100 * (1 - lines_n / best_n)
    RD = load_json("realdata.json")
    real_best = min(RD["OF + phase"]["fwhm"], RD["OF + phase + projectors"]["fwhm"])
    TE = load_json("timing_equalize.json")
    te_hi = TE[-1]
    n_max = max(int(k) for k in te_hi["after"])
    PH = load_json("phonon.json")
    def ph(name, e):
        r = PH[name]
        return min(r, key=lambda x: abs(x["E"] - e))["fwhm"]
    PHN = {"OF": "phOF", "OF + phase + projectors": "phProj", "Covariance filter": "phCov",
           "This work, label-free (science only)": "phFree",
           "This work, line-aware (calibration run)": "phCal",
           "This work, line-aware (calibration + science)": "phBoth"}
    ph_macros = {}
    for name, tag in PHN.items():
        for e, et in ((1000, "A"), (5000, "B"), (13750, "C"), (21250, "D"), (23750, "E")):
            ph_macros[tag + et] = fmt(ph(name, e), 0)
    ours_ph = np.array([x["fwhm"] for x in PH["This work, line-aware (calibration run)"]])
    cov_ph = np.array([x["fwhm"] for x in PH["Covariance filter"]])
    proj_ph = np.array([x["fwhm"] for x in PH["OF + phase + projectors"]])
    inside = np.array([x["E"] for x in PH["Covariance filter"]]) < 22200
    ph_macros["phGainCovMin"] = fmt(np.min(cov_ph[inside] / ours_ph[inside]), 1)
    ph_macros["phGainCovMax"] = fmt(np.max(cov_ph[inside] / ours_ph[inside]), 1)
    macros = dict(**ph_macros,
        realOF=fmt(RD["OF"]["fwhm"], 3), realPhase=fmt(RD["OF + phase"]["fwhm"], 3),
        realDecor=fmt(RD["OF + phase + projectors"]["fwhm"], 3),
        realStat=fmt(RD["ML stationary"]["fwhm"], 3), realOurs=fmt(RD["ML Woodbury (lines)"]["fwhm"], 3),
        realOursErr=fmt(RD["ML Woodbury (lines)"]["err"], 3),
        realLimit=fmt(RD["baseline_limit"]["fwhm"], 3),
        realImpBest=fmt(100 * (1 - RD["ML Woodbury (lines)"]["fwhm"] / real_best), 0),
        realImpOF=fmt(100 * (1 - RD["ML Woodbury (lines)"]["fwhm"] / RD["OF"]["fwhm"]), 0),
        realNtest=str(RD["n_test_line"]), realNtrain=str(RD["n_train_line"]),
        realOursDrift=fmt(RD["ML Woodbury (lines) + drift"]["fwhm"], 3),
        realOursDriftErr=fmt(RD["ML Woodbury (lines) + drift"]["err"], 3),
        realMassATS=fmt(RD["MASS (ATS + drift + phase)"]["fwhm"], 3),
        realMassATSErr=fmt(RD["MASS (ATS + drift + phase)"]["err"], 3),
        realMassLag=fmt(RD["MASS (5-lag + drift + phase)"]["fwhm"], 3),
        realMassSigma=fmt((RD["ML Woodbury (lines) + drift"]["fwhm"] - RD["MASS (ATS + drift + phase)"]["fwhm"])
                          / np.hypot(RD["ML Woodbury (lines) + drift"]["err"],
                                     RD["MASS (ATS + drift + phase)"]["err"]), 1),
        realOursEV=fmt(6 * RD["ML Woodbury (lines)"]["fwhm"], 2),
        realPhaseEV=fmt(6 * RD["OF + phase"]["fwhm"], 2),
        eqBefore=fmt(te_hi["jitter"] * 5e3, 1), eqAfter=fmt(te_hi["after"][str(n_max)] * 5e3, 1),
        eqCRB=fmt(te_hi["crb"] * 5e3, 1), eqNmax=f"{n_max:,}".replace(",", "\\,"),
        eqAfterLow=fmt(TE[0]["after"][str(n_max)] * 5e3, 1), eqBeforeLow=fmt(TE[0]["jitter"] * 5e3, 1),
        eqCRBLow=fmt(TE[0]["crb"] * 5e3, 1),
        eqAfterSmall=fmt(te_hi["after"][str(min(int(k) for k in te_hi["after"]))] * 5e3, 1),
        resPracticeHi=fmt(fw(R, "OF + phase + projectors")[-1]),
        resPhaseHiNoMorph=fmt(fw(Rn, "OF + phase")[-1]),
        resCovarHi=fmt(fw(R, "Covariance filter")[-1]),
        resCovarHiNoMorph=fmt(fw(Rn, "Covariance filter")[-1]),
        impBestLo=fmt(imp_m.min(), 0), impBestHi=fmt(imp_m.max(), 0),
        impBestLoNoMorph=fmt(imp_n.min(), 0), impBestHiNoMorph=fmt(imp_n.max(), 0),
        impBestTopMorph=fmt(imp_m[-1], 0), impBestTopNoMorph=fmt(imp_n[-1], 0),
        resLinesLo=fmt(lines.min()), resLinesHi=fmt(lines.max()),
        resOFLo=fmt(of.min()), resOFHi=fmt(of.max()),
        resStatHi=fmt(stat[-1]),
        oracleGap=fmt(np.max(np.abs(lines - orc)), 2),
        gainLo=fmt(gain[0], 1) + r"$\times$", gainMid=fmt(gain[i65], 1) + r"$\times$",
        gainHi=fmt(gain[-1], 1) + r"$\times$",
        gainMaxNoMorph=fmt(100 * (gain_n.max() - 1), 0) + r"\%",
        resOFHiNoMorph=fmt(of_n[-1]), resLinesHiNoMorph=fmt(lines_n[-1]),
        resStatHiNoMorph=fmt(fw(Rn, "ML stationary")[-1]),
        simAlpha=fmt(sim.alpha, 0), simLoop=fmt(sim.loop_gain, 1),
        simTau=fmt(sim.tau_eff * 1e6, 0), simIzero=fmt(sim.I0 * 1e6, 1),
        simPeakDrop=fmt(peaks[1000.] / peaks[12000.], 1),
        sipcaIters=str(m["sipca_iters"]), smax=fmt(M["man"].s_max_, 0),
        jitterTyp=fmt(np.median(jit) * 5e3, 0),
        jitterRatioNoMorph=fmt(np.median(jit_n[-3:] / crb_t_n[-3:]), 1),
        jitterNoMorphLo=fmt(jit_n[1:].min() * 5e3, 0), jitterNoMorphHi=fmt(jit_n[1:].max() * 5e3, 0),
        jitterLowE=fmt(jit_n[0] * 5e3, 0), crbLowE=fmt(crb_n[0] * 5e3, 0),
        jitterPhaseHi=fmt(pn[-1] * 5e3, 1), crbHi=fmt(crb_n[-1] * 5e3, 1),
        jitterPhaseMaxRatio=fmt(np.max(pn / crb_n), 2),
        jitterMorphLo=fmt(jm.min(), 0), jitterMorphHi=fmt(jm.max(), 0),
        normBiasZero=fmt(norm0, 1), ofBiasZero=fmt(of0, 2), mlBiasZero=fmt(ml0, 2),
        mlSpreadZero=fmt(100 * (ml_sd_ratio - 1), 0) + r"\%",
        woodburyPrec=r"$10^{%d}$" % int(np.floor(np.log10(max(r["max_rel_diff"] for r in wb)))),
        woodburySpeedup=r"$%s$" % fmt(w1024["t_dense"] / w1024["t_eval"], 0),
        rateLines=fmt(m["rates"]["ML Woodbury (lines)"], 0),
        rateOF=fmt(m["rates"]["OF"], 0),
    )
    with open(os.path.join(RESULTS, "numbers.tex"), "w") as f:
        f.write("% generated by scripts/make_numbers.py -- do not edit\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in macros.items():
        print(k, v)


if __name__ == "__main__":
    main()
