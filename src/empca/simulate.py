"""Physical simulator for voltage-biased transition-edge sensor (TES) pulses.

The simulator integrates the coupled electro-thermal Langevin equations of a
one-body TES (Irwin & Hilton 2005) with the Euler--Maruyama method,
vectorised over events.  It produces, without any phenomenological shortcut,

* electrothermal saturation (the TES is driven up the transition and the pulse
  flattens and lengthens with energy),
* energy-dependent pulse shapes and decay times,
* position-dependent rise times (absorber thermalisation time ``tau_th``
  modulated by a morphology parameter ``eta``),
* continuous (sub-sample) arrival times,
* stationary baseline noise *and* genuinely non-stationary noise during the
  pulse, because the Johnson and thermal-fluctuation noise sources depend on
  the instantaneous resistance and temperature.

Units are SI throughout; the returned traces are the drop in TES current in
micro-amperes (positive-going pulses).
"""
from dataclasses import dataclass, field

import numpy as np

KB = 1.380649e-23
EV = 1.602176634e-19

__all__ = ["TESParameters", "TESSimulator", "SimulatedData", "PhononParameters", "PhononSimulator"]


@dataclass
class TESParameters:
    """Physical parameters of the simulated sensor and readout."""

    Tc: float = 0.100          # transition temperature [K]
    Tb: float = 0.055          # bath temperature [K]
    width: float = 1.4e-3      # tanh transition width [K]
    Rn: float = 10e-3          # normal resistance [Ohm]
    R0_frac: float = 0.30      # bias point R0/Rn
    RL: float = 0.3e-3         # shunt + parasitic resistance [Ohm]
    L: float = 40e-9           # SQUID input + stray inductance [H]
    C: float = 1.2e-12         # heat capacity [J/K]
    G: float = 250e-12         # thermal conductance at Tc [W/K]
    n: float = 3.0             # thermal conductance exponent
    F_link: float = 0.6        # thermal-link noise factor
    squid_noise: float = 6e-12  # SQUID current noise [A/sqrt(Hz)]
    tau_th0: float = 12e-6     # absorber thermalisation time at eta = 0 [s]
    eta_scale: float = 0.02    # tau_th = tau_th0 * (1 + eta_scale * eta)
    fs: float = 200e3          # output sampling rate [Hz]
    oversample: int = 8        # Euler-Maruyama sub-steps per output sample
    n_samples: int = 1024      # record length
    pretrigger: int = 256      # nominal arrival sample
    burn_in: float = 1e-3      # noise burn-in before each record [s]

    @property
    def K(self):
        return self.G / (self.n * self.Tc ** (self.n - 1))


@dataclass
class SimulatedData:
    traces: np.ndarray
    energy: np.ndarray          # true energy [eV]
    eta: np.ndarray             # true morphology parameter
    t0: np.ndarray              # true arrival time [samples]
    labels: np.ndarray = field(default=None)  # line index (-1: continuum)


class TESSimulator:
    """Euler--Maruyama integration of the TES electro-thermal equations.

    Parameters
    ----------
    params : TESParameters, optional
    seed : int or numpy Generator, optional
    nonstationary : bool
        If False the Johnson and phonon noise amplitudes are frozen at their
        bias-point values, giving purely stationary (additive) noise.  Useful
        for controlled ablations.
    """

    def __init__(self, params=None, seed=None, nonstationary=True):
        self.p = params if params is not None else TESParameters()
        self.rng = np.random.default_rng(seed)
        self.nonstationary = nonstationary
        self._bias_point()

    # ------------------------------------------------------------------ model
    def resistance(self, T):
        p = self.p
        return 0.5 * p.Rn * (1.0 + np.tanh((T - p.Tc) / p.width))

    def _bias_point(self):
        p = self.p
        r = 2 * p.R0_frac - 1
        self.T0 = p.Tc + p.width * np.arctanh(r)
        self.R0 = self.resistance(self.T0)
        P0 = p.K * (self.T0 ** p.n - p.Tb ** p.n)
        self.I0 = np.sqrt(P0 / self.R0)
        self.Vb = self.I0 * (self.R0 + p.RL)
        dRdT = 0.5 * p.Rn / p.width / np.cosh((self.T0 - p.Tc) / p.width) ** 2
        self.alpha = self.T0 / self.R0 * dRdT
        G0 = p.n * p.K * self.T0 ** (p.n - 1)
        self.loop_gain = P0 * self.alpha / (G0 * self.T0)
        self.tau_eff = p.C / G0 / (1 + self.loop_gain)

    def _deposit(self, t_lo, t_hi, t0, E, tau):
        """Energy deposited in [t_lo, t_hi) by an exponential thermalisation."""
        a = np.clip(t_lo - t0, 0.0, None)
        b = np.clip(t_hi - t0, 0.0, None)
        return E * (np.exp(-a / tau) - np.exp(-b / tau))

    # -------------------------------------------------------------- simulate
    def simulate(self, energy, eta=None, t0=None, batch=4000):
        """Simulate records.

        Parameters
        ----------
        energy : array_like
            Deposited energies in eV (0 gives a pure noise record).
        eta : array_like, optional
            Morphology (position) parameter, default 0.
        t0 : array_like, optional
            Arrival time in output samples (continuous); default ``pretrigger``.

        Returns
        -------
        traces : ndarray, shape (n_events, n_samples), micro-amperes.
        """
        energy = np.atleast_1d(np.asarray(energy, float))
        n_ev = energy.size
        eta = np.zeros(n_ev) if eta is None else np.broadcast_to(eta, (n_ev,)).astype(float)
        t0 = np.full(n_ev, float(self.p.pretrigger)) if t0 is None else np.broadcast_to(t0, (n_ev,)).astype(float)
        out = np.empty((n_ev, self.p.n_samples))
        for lo in range(0, n_ev, batch):
            sl = slice(lo, lo + batch)
            out[sl] = self._simulate_batch(energy[sl], eta[sl], t0[sl])
        return out

    def _simulate_batch(self, energy, eta, t0):
        p, rng = self.p, self.rng
        m = energy.size
        dt_s = 1.0 / p.fs
        dt = dt_s / p.oversample
        n_burn = int(round(p.burn_in / dt))
        n_fine = p.n_samples * p.oversample

        E = energy * EV
        tau = p.tau_th0 * (1.0 + p.eta_scale * eta)
        t_arr = t0 * dt_s  # arrival time measured from record start

        T = np.full(m, self.T0)
        I = np.full(m, self.I0)
        acc = np.zeros(m)
        out = np.empty((m, p.n_samples))
        sq = np.sqrt(dt)
        T_frozen, R_frozen = self.T0, self.R0

        for k in range(-n_burn, n_fine):
            R = self.resistance(T)
            Tn, Rn_ = (T, R) if self.nonstationary else (T_frozen, R_frozen)
            # noise sources: one-sided PSDs S, increments sqrt(S/2 * dt) * N(0,1)
            vJ = np.sqrt(2 * KB * Tn * Rn_) * rng.standard_normal(m) / sq
            vL = np.sqrt(2 * KB * p.Tb * p.RL) * rng.standard_normal(m) / sq
            Gt = p.n * p.K * Tn ** (p.n - 1)
            pph = np.sqrt(2 * KB * Tn ** 2 * Gt * p.F_link) * rng.standard_normal(m) / sq

            Pb = p.K * (T ** p.n - p.Tb ** p.n)
            dT = (I * I * R - Pb + pph - I * vJ) * dt / p.C
            if k >= 0:
                t = k * dt
                dT += self._deposit(t, t + dt, t_arr, E, tau) / p.C
            dI = (self.Vb - I * (R + p.RL) + vJ + vL) * dt / p.L
            T = T + dT
            I = I + dI
            if k >= 0:
                acc += I
                if (k + 1) % p.oversample == 0:
                    out[:, k // p.oversample] = acc / p.oversample
                    acc[:] = 0.0

        # SQUID white noise at the output sampling rate
        out += p.squid_noise * np.sqrt(p.fs / 2) * rng.standard_normal(out.shape)
        return (self.I0 - out) * 1e6

    def noise(self, n_records):
        """Pure noise records (energy 0)."""
        return self.simulate(np.zeros(n_records))

    # --------------------------------------------------------------- spectra
    def spectrum(self, n_events, lines=(1500., 3000., 4500., 5900., 8000., 10000., 12000.),
                 line_fraction=0.6, e_range=(500., 13000.), eta_range=(-1.0, 1.0),
                 jitter=2.0):
        """Simulate an unlabelled spectrum: mono-energetic lines on a flat continuum.

        Returns a :class:`SimulatedData` with true parameters attached.
        """
        rng = self.rng
        lines = np.asarray(lines, float)
        n_line = int(round(line_fraction * n_events))
        labels = np.full(n_events, -1)
        labels[:n_line] = rng.integers(0, lines.size, n_line)
        energy = rng.uniform(*e_range, n_events)
        energy[:n_line] = lines[labels[:n_line]]
        perm = rng.permutation(n_events)
        energy, labels = energy[perm], labels[perm]
        eta = rng.uniform(*eta_range, n_events)
        t0 = self.p.pretrigger + rng.uniform(-jitter, jitter, n_events)
        traces = self.simulate(energy, eta, t0)
        return SimulatedData(traces, energy, eta, t0, labels)


# ---------------------------------------------------------------------------
# Phonon-mediated detector (dark-matter style), phenomenological two-component
# ---------------------------------------------------------------------------
@dataclass
class PhononParameters(TESParameters):
    """A TES read out through a crystal absorber.

    An energy ``E`` deposited at position ``(rho^2, z)`` in the crystal (``rho^2``
    uniform in [0, 1] for a uniform disk, ``z`` the normalised distance from the
    instrumented face) produces

    * an athermal phonon signal collected by the TES with efficiency
      ``eps = eps0 (1 - k_rho rho^2)(1 - k_z_eps z)`` and collection time
      ``tau_a = tau_a0 (1 + k_z_tau z)`` (fast, position dependent), and
    * a thermal component: the remaining ``(1 - eps) E`` heats the crystal,
      which relaxes through the TES link ``G_ab`` and the bath link ``G_cb``
      (slow), in the spirit of the two-component model of Proebst et al. (1995).
    """

    eps0: float = 0.25
    k_rho: float = 0.15
    k_z_eps: float = 0.10
    tau_a0: float = 20e-6
    k_z_tau: float = 1.0
    C_abs: float = 1.6e-12
    G_ab: float = 50e-12
    G_cb: float = 1.0e-9


class PhononSimulator(TESSimulator):
    """Euler--Maruyama simulation of a TES coupled to a crystal absorber."""

    def __init__(self, params=None, seed=None, nonstationary=True):
        super().__init__(params if params is not None else PhononParameters(), seed, nonstationary)

    def _bias_point(self):
        p = self.p
        r = 2 * p.R0_frac - 1
        self.T0 = p.Tc + p.width * np.arctanh(r)
        self.R0 = self.resistance(self.T0)
        self.Ta0 = (p.G_ab * self.T0 + p.G_cb * p.Tb) / (p.G_ab + p.G_cb)
        P0 = p.K * (self.T0 ** p.n - p.Tb ** p.n) + p.G_ab * (self.T0 - self.Ta0)
        self.I0 = np.sqrt(P0 / self.R0)
        self.Vb = self.I0 * (self.R0 + p.RL)
        dRdT = 0.5 * p.Rn / p.width / np.cosh((self.T0 - p.Tc) / p.width) ** 2
        self.alpha = self.T0 / self.R0 * dRdT
        G0 = p.n * p.K * self.T0 ** (p.n - 1) + p.G_ab
        self.loop_gain = P0 * self.alpha / (G0 * self.T0)
        self.tau_eff = p.C / G0 / (1 + self.loop_gain)

    def collection(self, rho2, z):
        p = self.p
        eps = p.eps0 * (1 - p.k_rho * rho2) * (1 - p.k_z_eps * z)
        return eps, p.tau_a0 * (1 + p.k_z_tau * z)

    def simulate(self, energy, rho2=None, z=None, t0=None, batch=4000):
        energy = np.atleast_1d(np.asarray(energy, float))
        n_ev = energy.size
        rho2 = np.zeros(n_ev) if rho2 is None else np.broadcast_to(rho2, (n_ev,)).astype(float)
        z = np.zeros(n_ev) if z is None else np.broadcast_to(z, (n_ev,)).astype(float)
        t0 = np.full(n_ev, float(self.p.pretrigger)) if t0 is None else np.broadcast_to(t0, (n_ev,)).astype(float)
        out = np.empty((n_ev, self.p.n_samples))
        for lo in range(0, n_ev, batch):
            sl = slice(lo, lo + batch)
            out[sl] = self._simulate_phonon_batch(energy[sl], rho2[sl], z[sl], t0[sl])
        return out

    def _simulate_phonon_batch(self, energy, rho2, z, t0):
        p, rng = self.p, self.rng
        m = energy.size
        dt_s = 1.0 / p.fs
        dt = dt_s / p.oversample
        n_burn = int(round(p.burn_in / dt))
        n_fine = p.n_samples * p.oversample
        eps, tau = self.collection(rho2, z)
        E = energy * EV
        t_arr = t0 * dt_s
        k_dep = np.floor(t_arr / dt).astype(int)        # step receiving the thermal deposit
        T = np.full(m, self.T0)
        Ta = np.full(m, self.Ta0)
        I = np.full(m, self.I0)
        acc = np.zeros(m)
        out = np.empty((m, p.n_samples))
        sq = np.sqrt(dt)
        for k in range(-n_burn, n_fine):
            R = self.resistance(T)
            Tn, Rn_ = (T, R) if self.nonstationary else (self.T0, self.R0)
            vJ = np.sqrt(2 * KB * Tn * Rn_) * rng.standard_normal(m) / sq
            vL = np.sqrt(2 * KB * p.Tb * p.RL) * rng.standard_normal(m) / sq
            Gt = p.n * p.K * Tn ** (p.n - 1)
            pph = np.sqrt(2 * KB * Tn ** 2 * Gt * p.F_link) * rng.standard_normal(m) / sq
            pab = np.sqrt(2 * KB * Tn ** 2 * p.G_ab) * rng.standard_normal(m) / sq
            pcb = np.sqrt(2 * KB * Ta ** 2 * p.G_cb) * rng.standard_normal(m) / sq
            Pb = p.K * (T ** p.n - p.Tb ** p.n)
            Pab = p.G_ab * (T - Ta)
            dT = (I * I * R - Pb - Pab + pph - pab - I * vJ) * dt / p.C
            dTa = (Pab - p.G_cb * (Ta - p.Tb) + pab - pcb) * dt / p.C_abs
            if k >= 0:
                t = k * dt
                dT += self._deposit(t, t + dt, t_arr, eps * E, tau) / p.C
                dTa += np.where(k == k_dep, (1 - eps) * E, 0.0) / p.C_abs
            dI = (self.Vb - I * (R + p.RL) + vJ + vL) * dt / p.L
            T, Ta, I = T + dT, Ta + dTa, I + dI
            if k >= 0:
                acc += I
                if (k + 1) % p.oversample == 0:
                    out[:, k // p.oversample] = acc / p.oversample
                    acc[:] = 0.0
        out += p.squid_noise * np.sqrt(p.fs / 2) * rng.standard_normal(out.shape)
        return (self.I0 - out) * 1e6

    def _positions(self, n):
        return self.rng.uniform(0, 1, n), self.rng.uniform(0, 1, n)

    def calibration_run(self, n_events, lines=(1300., 5900., 10400., 14400., 17500., 22200.),
                        line_fraction=0.7, e_range=(500., 25000.), jitter=2.0):
        """Calibration data: mono-energetic lines on a Compton-like flat continuum."""
        rng = self.rng
        lines = np.asarray(lines, float)
        n_line = int(round(line_fraction * n_events))
        labels = np.full(n_events, -1)
        labels[:n_line] = rng.integers(0, lines.size, n_line)
        energy = rng.uniform(*e_range, n_events)
        energy[:n_line] = lines[labels[:n_line]]
        perm = rng.permutation(n_events)
        energy, labels = energy[perm], labels[perm]
        rho2, z = self._positions(n_events)
        t0 = self.p.pretrigger + rng.uniform(-jitter, jitter, n_events)
        traces = self.simulate(energy, rho2, z, t0)
        return SimulatedData(traces, energy, np.stack([rho2, z], 1), t0, labels)

    def science_run(self, n_events, e_range=(500., 25000.), exp_fraction=0.5, exp_scale=3000.,
                    jitter=2.0):
        """Science data without lines: flat plus exponentially falling background."""
        rng = self.rng
        n_exp = int(round(exp_fraction * n_events))
        e_exp = e_range[0] + rng.exponential(exp_scale, 4 * n_exp)
        e_exp = e_exp[e_exp < e_range[1]][:n_exp]
        energy = np.r_[e_exp, rng.uniform(*e_range, n_events - e_exp.size)]
        energy = rng.permutation(energy)
        rho2, z = self._positions(n_events)
        t0 = self.p.pretrigger + rng.uniform(-jitter, jitter, n_events)
        traces = self.simulate(energy, rho2, z, t0)
        return SimulatedData(traces, energy, np.stack([rho2, z], 1), t0, np.full(n_events, -1))
