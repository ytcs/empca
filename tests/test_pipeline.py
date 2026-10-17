import numpy as np

from empca import (EnergyCalibration, OptimalFilter, PulseManifold, PulseProcessor,
                   ShiftInvariantPCA, StationaryNoise, find_clusters, line_resolution)
from empca.shifts import shift


def test_sipca_recovers_arrival_times(toy_data, noise_model):
    p = ShiftInvariantPCA(3, max_shift=6, snr_min=20).fit(toy_data["traces"], noise_model)
    dt = p.t0_ - toy_data["t0"]
    big = toy_data["E"] > 3
    resid = dt[big] - np.median(dt[big])
    assert np.std(resid) < 0.15
    # objective is non-decreasing
    obj = np.array(p.objective_)
    assert np.all(np.diff(obj) >= -1e-9 * obj[1:])
    # basis is orthonormal
    assert np.allclose(p.basis_ @ p.basis_.T, np.eye(3), atol=1e-8)


def test_manifold_is_monotone_and_unbiased_at_zero(toy_data, noise_model):
    p = ShiftInvariantPCA(3, max_shift=6, snr_min=20).fit(toy_data["traces"], noise_model)
    m = PulseManifold().fit(p.coeffs_)
    E, s = toy_data["E"], m.s_
    order = np.argsort(E)
    med = [np.median(s[E == e]) for e in toy_data["lines"]]
    assert np.all(np.diff(med) > 0)
    # arc-length parameterisation: unit speed
    grid = np.linspace(0.05, 0.95, 20) * m.s_max_
    assert np.allclose(m.speed(grid), 1.0, atol=0.02)
    # pure noise projects to a zero-mean coordinate (no rectification)
    a0 = np.random.default_rng(3).standard_normal((20000, 3))
    assert abs(np.mean(m.project(a0))) < 0.05


def test_processor_end_to_end(toy_data, noise_model):
    X, E, lines = toy_data["traces"], toy_data["E"], toy_data["lines"]
    for cov in (None, "curve", "lines"):
        proc = PulseProcessor(n_components=3, max_shift=6, snr_min=20, n_lines=4,
                              n_noise_modes=4, covariance=cov).fit(X, toy_data["noise"])
        res = proc.transform(X)
        assert len(res) == len(X) and np.all(np.isfinite(res.nll))
        proc.calibrate(proc.line_centers_, lines)
        e_new = proc.energy(res.s)
        # calibrated line centroids are recovered
        for e in lines:
            assert abs(np.median(e_new[E == e]) - e) < 0.25 * line_resolution(e_new[E == e])[1]
    # in the linear regime the manifold estimator is as good as the optimal filter
    of = OptimalFilter(max_shift=6).fit(X[E == 1.0], proc.noise_)
    A, _ = of.transform(X)
    e_of = A * 1.0 / np.median(A[E == 1.0])
    w_of = line_resolution(e_of[E == 1.0])[1]
    w_new = line_resolution(e_new[E == 1.0])[1]
    assert w_new < 1.1 * w_of
    # templates come back in raw units (up to the unobservable DC offset) and
    # their pulse height matches the true one
    tmpl = proc.template(np.median(res.s[E == 6.0]))[0]
    height = tmpl.max() - np.median(tmpl[:40])
    assert abs(height - np.median(toy_data["clean"][E == 6.0].max(axis=1))) < 0.05 * height
    # Cramer-Rao bound of the learned model is finite and positive
    sig = proc.crb(np.median(res.s[E == 6.0]))
    assert np.all(np.isfinite(sig)) and np.all(sig > 0)
