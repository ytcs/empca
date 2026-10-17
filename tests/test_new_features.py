import numpy as np

from empca import PhaseEqualizer, PulseManifold


def test_phase_equalizer_removes_periodic_bias():
    rng = np.random.default_rng(0)
    n = 40000
    t0 = rng.uniform(-2, 2, n)
    phi = np.mod(t0, 1)
    tau = t0 + 0.3 + 0.04 * np.sin(2 * np.pi * phi) + 0.002 * rng.standard_normal(n)
    learn = np.arange(n) < n // 2
    g = np.where(learn, 0, -1)
    eq = PhaseEqualizer(3).fit(tau, g)
    corr = eq.transform(tau[~learn], np.zeros((~learn).sum(), int))
    before = np.std(tau[~learn] - t0[~learn])
    after = np.std(corr - t0[~learn])
    assert after < 0.25 * before
    assert eq.required_events(0.004) > 1e4


def test_manifold_uses_secant_for_single_line():
    rng = np.random.default_rng(1)
    centre = np.array([100.0, 30.0, 0.0])
    # one line whose internal scatter is dominated by a transverse direction
    line = centre + rng.standard_normal((3000, 3)) * [1, 1, 1] + \
        np.outer(rng.standard_normal(3000), [0, 0, 15.0])
    tail = np.outer(rng.uniform(0.3, 0.95, 50), centre) + rng.standard_normal((50, 3))
    A = np.vstack([line, tail])
    groups = np.r_[np.zeros(3000, int), -np.ones(50, int)]
    m = PulseManifold().fit(A, groups=groups)
    assert m.mode_ == "secant"
    t = m.curve([m.s_[:3000].mean()], 1)[0]
    assert abs(t @ centre / np.linalg.norm(centre)) > 0.99
    # without the cluster information the curve follows the transverse scatter more
    m0 = PulseManifold().fit(A)
    s_line = m.s_[:3000]
    assert np.std(s_line) < np.std(m0.s_[:3000])
