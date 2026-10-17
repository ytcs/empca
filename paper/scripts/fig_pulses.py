"""Figure: simulated TES pulses -- saturation, shape evolution, noise spectrum and
non-stationary noise during the pulse."""
import numpy as np

from common import FIG, mpl, simulate
from empca import StationaryNoise
from empca.simulate import TESParameters, TESSimulator

plt = mpl()
p = TESParameters()
fs = p.fs
t_us = (np.arange(p.n_samples) - p.pretrigger) / fs * 1e6

fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
energies = [1000., 3000., 6000., 9000., 12000.]
cmap = plt.get_cmap("viridis")
sim = TESSimulator(p, seed=11)
means = {}
for i, e in enumerate(energies):
    x = sim.simulate(np.full(200, e)).mean(axis=0)
    means[e] = x
    ax[0].plot(t_us, x, color=cmap(i / (len(energies) - 1)), label=f"{e / 1e3:g} keV")
    ax[1].plot(t_us, x / x.max(), color=cmap(i / (len(energies) - 1)))
ax[0].set_xlim(-100, 2000)
ax[1].set_xlim(-100, 2000)
ax[0].set_xlabel(r"$t - t_0$ [$\mu$s]")
ax[0].set_ylabel(r"$-\Delta I$ [$\mu$A]")
ax[0].legend(frameon=False, fontsize=5.5)
ax[1].set_xlabel(r"$t - t_0$ [$\mu$s]")
ax[1].set_ylabel("normalised")
ax[0].set_title("(a) pulses", fontsize=8)
ax[1].set_title("(b) shape evolution", fontsize=8)

D = simulate("morph")
noise = StationaryNoise.from_records(D["noise"])
f = np.fft.rfftfreq(p.n_samples, 1 / fs)
ax[2].loglog(f[1:-1], np.sqrt(noise.psd(fs)[1:-1]) * 1e6, color="k")
from matplotlib.ticker import ScalarFormatter, NullFormatter
ax[2].yaxis.set_major_formatter(ScalarFormatter())
ax[2].yaxis.set_minor_formatter(NullFormatter())
ax[2].set_yticks([10, 20, 40, 60])
ax[2].set_xlabel("frequency [Hz]")
ax[2].set_ylabel(r"noise [pA/$\sqrt{\rm Hz}$]")
ax[2].set_title("(c) baseline noise", fontsize=8)

for flag, ls, lab in ((True, "-", "physical (non-stationary)"), (False, ":", "frozen sources")):
    s2 = TESSimulator(p, seed=12, nonstationary=flag)
    for e, col in ((3000., cmap(0.25)), (12000., cmap(1.0))):
        x = s2.simulate(np.full(1500, e))
        v = (x - x.mean(0)).var(axis=0)
        v0 = v[: p.pretrigger - 10].mean()
        k = np.ones(9) / 9
        ax[3].plot(t_us, np.convolve(v / v0, k, mode="same"), color=col, ls=ls,
                   label=f"{e / 1e3:g} keV, {lab}")
ax[3].axhline(1, color="grey", lw=0.6)
ax[3].set_xlim(-100, 2000)
ax[3].set_xlabel(r"$t - t_0$ [$\mu$s]")
ax[3].set_ylabel("noise variance / baseline")
ax[3].set_title("(d) non-stationary noise", fontsize=8)
ax[3].legend(frameon=False, fontsize=5)
fig.tight_layout(w_pad=0.6)
fig.savefig(f"{FIG}/pulses.pdf")

# numbers quoted in the text
sim0 = TESSimulator(p)
print(f"alpha={sim0.alpha:.1f} L={sim0.loop_gain:.1f} tau_eff={sim0.tau_eff*1e6:.0f}us "
      f"I0={sim0.I0*1e6:.1f}uA R0={sim0.R0*1e3:.1f}mOhm")
for e, x in means.items():
    print(e, "peak/E [nA/eV]", x.max() / e * 1e3)
