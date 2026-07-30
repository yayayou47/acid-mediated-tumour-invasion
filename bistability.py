"""
bistability.py -- Numerical illustration of the bistable regime
(alpha > 1, delta > delta_c), where the coexistence equilibrium E3 is a
SADDLE and its stable manifold separates the basins of E1 and E2.

This is the region that the corrected existence condition of
Proposition 3.5 opens up, and which none of the three phenotypes of
Table 2 visits.  Two runs are performed with IDENTICAL parameters and
differing only in the initial tumour burden:

    * a small seed  -> the tissue clears the tumour, u -> E1 = (1, 0, 0)
    * a large seed  -> the tumour invades,          u -> E2 = (0, 1, g/w)

which is the operational signature of bistability: the outcome is
decided by the initial condition, not by the parameters.

Produces `bistability.pdf`.

Author: Yaya Youssouf Yaya
Date:   2026-07-11
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from main import Grid2D, Parameters, Solver
from params import SEED

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
np.random.seed(SEED)

# ─── Bistable parameter set: alpha > 1 AND delta > delta_c ───────────
ALPHA, BETA, GAMMA, OMEGA, DELTA = 1.2, 1.5, 5.0, 8.0, 2.5
DELTA1 = DELTA2 = 5e-3
DELTA_C = OMEGA / GAMMA                      # 1.6  ->  delta/delta_c = 1.5625

NX = NY = 100
LX = LY = 2.0
DT = 5e-3
T_FINAL = 40.0

C_N = "#0072B2"     # normal cells
C_T = "#D55E00"     # tumour cells


SEED_AMP = 0.95          # tumour density inside the seed


def initial_data(grid, radius: float):
    """Healthy tissue at carrying capacity, circular tumour seed at centre.

    Inside the disc of the given radius the tumour is at density SEED_AMP and
    the tissue is correspondingly displaced; outside, tissue is at carrying
    capacity and there is no tumour.  The initial excess acid is zero.
    """
    X, Y = grid.X, grid.Y
    disc = (np.sqrt((X - LX / 2) ** 2 + (Y - LY / 2) ** 2) <= radius)
    T0 = SEED_AMP * disc.astype(float)
    N0 = 1.0 - SEED_AMP * disc.astype(float)
    H0 = np.zeros_like(T0)
    return N0, T0, H0


def run(radius: float):
    grid = Grid2D(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    p = Parameters(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                   omega=OMEGA, delta1=DELTA1, delta2=DELTA2)
    solver = Solver(grid, p, dt=DT, enforce_clip=False)
    N, T, H = initial_data(grid, radius)

    nsteps = int(T_FINAL / DT)
    ts, mN, mT = [], [], []
    for n in range(nsteps + 1):
        if n % 20 == 0:
            ts.append(n * DT)
            mN.append(float(N.mean()))
            mT.append(float(T.mean()))
        if n == nsteps:
            break
        N, T, H = solver.step(N, T, H)
    return (np.array(ts), np.array(mN), np.array(mT),
            solver.violation_report(), p)


def main() -> None:
    p_probe = Parameters(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                         omega=OMEGA, delta1=DELTA1, delta2=DELTA2)
    e3 = p_probe.E3
    print(f"regime = {p_probe.regime}, delta/delta_c = {DELTA/DELTA_C:.4f}")
    print(f"E3 = {e3},  saddle = {p_probe.E3_is_saddle}")

    # The critical radius lies between 0.55 and 0.60 on this domain
    # (bracketed by bistability_sweep.py; mesh-independent Nx=100,140).
    small = run(radius=0.35)     # sub-threshold seed
    large = run(radius=0.65)     # supra-threshold seed

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    titles = [r"(a) seed radius $0.35$ $\rightarrow$ tumour cleared ($E_1$)",
              r"(b) seed radius $0.65$ $\rightarrow$ full invasion ($E_2$)"]

    for ax, (ts, mN, mT, viol, p), title in zip(axes, (small, large), titles):
        ax.plot(ts, mN, color=C_N, lw=2.2, label=r"$\langle N \rangle$")
        ax.plot(ts, mT, color=C_T, lw=2.2, label=r"$\langle T \rangle$")
        ax.axhline(1.0, color="k", lw=0.9, ls=(0, (1, 2)), alpha=0.6)
        ax.axhline(0.0, color="k", lw=0.9, ls=(0, (1, 2)), alpha=0.6)
        if e3 is not None:
            ax.axhline(e3[0], color=C_N, lw=1.1, ls=(0, (5, 2)), alpha=0.75)
            ax.axhline(e3[1], color=C_T, lw=1.1, ls=(0, (5, 2)), alpha=0.75)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel(r"time $t$")
        ax.set_xlim(0, T_FINAL)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("spatial mean density")
    axes[0].legend(loc="center right", frameon=False, fontsize=10)
    axes[1].annotate("saddle $E_3$", xy=(T_FINAL * 0.72, e3[1]),
                     xytext=(T_FINAL * 0.55, 0.52), fontsize=9,
                     color="#444",
                     arrowprops=dict(arrowstyle="-", lw=0.8, color="#444"))

    fig.suptitle(
        rf"Bistable regime  $\alpha = {ALPHA} > 1$, "
        rf"$\delta/\delta_c = {DELTA/DELTA_C:.2f} > 1$: "
        r"identical parameters, opposite outcomes",
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig("bistability.pdf", bbox_inches="tight")
    print("wrote bistability.pdf")

    for name, (_, mN, mT, viol, _) in (("small", small), ("large", large)):
        worst = max(v for k, v in viol.items() if k != "n_steps")
        print(f"  {name} seed: final <N> = {mN[-1]:.4f}, "
              f"<T> = {mT[-1]:.4f}, worst box violation = {worst:.2e}")


if __name__ == "__main__":
    main()
