import numpy as np
import pytest

from empca.noise import StationaryNoise
from empca.shifts import shift

N = 256


def pulse(E, n=N, t0=64.0, sat=6.0, eta=0.0):
    """Toy saturating pulse: amplitude ~ sat*tanh(E/sat), decay slows with E,
    rise time modulated by the morphology parameter ``eta``."""
    t = np.arange(n) - t0
    tr, tf = 2.0 * (1 + 0.1 * eta), 12.0 * (1 + 0.15 * E / sat)
    p = np.where(t > 0, np.exp(-np.clip(t, 0, None) / tf) - np.exp(-np.clip(t, 0, None) / tr), 0.0)
    p /= p.max()
    return sat * np.tanh(E / sat) * p


@pytest.fixture(scope="session")
def noise_model():
    f = np.fft.rfftfreq(N)
    lam = 0.02 * (1.0 + (0.05 / np.maximum(f, 1e-3)) ** 1.0 + 1.0 / (1 + (f / 0.05) ** 2))
    return StationaryNoise(lam, N)


@pytest.fixture(scope="session")
def toy_data(noise_model):
    rng = np.random.default_rng(0)
    n = 3000
    lines = np.array([1.0, 3.0, 6.0, 9.0])
    E = np.where(rng.random(n) < 0.6, lines[rng.integers(0, 4, n)], rng.uniform(0.3, 10.0, n))
    t0 = rng.uniform(-2, 2, n)
    eta = rng.uniform(-1, 1, n)
    clean = np.stack([pulse(e, eta=h) for e, h in zip(E, eta)])
    traces = shift(clean, t0) + noise_model.sample(n, rng)
    noise = noise_model.sample(2000, rng)
    return dict(traces=traces, E=E, t0=t0, lines=lines, noise=noise, clean=clean)
