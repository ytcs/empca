import numpy as np

from empca.simulate import TESSimulator, TESParameters


def test_bias_point_and_saturation():
    sim = TESSimulator(TESParameters(n_samples=256, pretrigger=64), seed=0)
    assert np.isclose(sim.R0, 0.3 * sim.p.Rn)
    assert sim.loop_gain > 10
    x = sim.simulate([2000.0, 12000.0])
    # saturating: peak per unit energy drops at high energy
    assert x[1].max() / 12000 < 0.75 * x[0].max() / 2000
    # baseline is flat before the arrival time
    assert abs(x[:, :50].mean()) < 0.05


def test_noise_records_are_stationary():
    sim = TESSimulator(TESParameters(n_samples=256, pretrigger=64), seed=1)
    n = sim.noise(400)
    v = n.var(axis=0)
    assert v[:128].mean() / v[128:].mean() - 1 < 0.15
