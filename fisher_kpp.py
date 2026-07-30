"""
fisher_kpp.py -- Travelling front speed of the aggressive phenotype,
measured properly.

The first submitted version reported c_num/c* ~ 0.93, i.e. a front slower
than the KPP minimal speed -- which cannot be right, since c* is a lower
bound for fronts issued from steep (here, compactly supported) data.
The deficit had two causes, and neither was physical:

  1. UNDER-RESOLUTION.  The wing of the front has width
     sqrt(delta2/beta) ~ 0.058, and the old grid (Nx = 200 on [0,4],
     h ~ 0.020) resolved it with barely 3 nodes.

  2. TRANSIENT.  The speed was fitted on t in [4, 12].  A pulled KPP front
     approaches its asymptotic speed only algebraically slowly -- the
     Bramson logarithmic delay, x_f(t) = c* t - (3 / 2 lambda*) ln t + O(1)
     with lambda* = sqrt(r/D) -- so an early-time fit necessarily
     UNDER-estimates c*.

With h = 0.005 (about 12 nodes across the wing) and the speed measured on
successive late windows, the ratio rises to c* and slightly past it (1.004 on
[55,75]).  That small excess survives every check:

  * not discretisation -- halving h and dt leaves it unchanged
    (1.0037 -> 1.0036 on [55,75]);
  * not the Bramson tail, which is a DEFICIT (instantaneous speed
    c* - 3/(2 lambda* t) < c*) and can only depress the measurement;
  * not a transient -- `--longtime` carries the run to t = 250 on a
    domain lengthened to [0,42] and the ratio RISES and saturates:
    1.0037, 1.0046, 1.0050, 1.0052, 1.0053, 1.0053 on the six windows
    from [55,75] to [210,250].  The first value reproduces the short
    domain, so the excess is domain-independent too.

The front is therefore weakly PUSHED, by 0.53%.  `--edge` shows why.
Far ahead the state is exactly (1,0,0), so c* = 2 sqrt(delta2 beta (1-alpha))
IS the linear spreading speed; but acid generated in the front diffuses ahead
and depresses N, so the growth rate beta(1-alpha N) felt inside the front
exceeds the nominal beta(1-alpha) -- by 11.7% at T = 1e-1, 4.1% at 1e-3, 1.1%
at 1e-6, decaying to 0 as T -> 0 and N -> 1.  Vanishing at the leading edge,
it cannot change the linear speed; substantial in the interior, it makes the
nonlinear front outrun its own linearisation.  Agreement with c* is better
than 1%, on the side the minimal-speed principle requires.

Produces `fisher_kpp_speed.pdf` (default), `fisher_kpp_longtime.npz`
(`--longtime`), `fisher_kpp_edge.npz` (`--edge`).

Author: Yaya Youssouf Yaya
Date:   2026-07-13
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from main import Grid2D, Parameters, Solver
from params import AGGRESSIVE

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                     "axes.labelsize": 12, "figure.dpi": 150})

# ─── Quasi-1D planar-front geometry ──────────────────────────────────
LX, NX = 14.0, 2801           # h = 0.005  ->  ~12 nodes across the wing
LY, NY = 0.1, 4
DT = 5e-4
T_END = 80.0
X_SEED = 0.15
LEVEL = 0.5                   # front = half-amplitude level set

# windows on which the apparent speed is measured
WINDOWS = [(10, 20), (20, 30), (30, 40), (40, 55), (55, 75)]
FIT_WINDOW = (55, 75)         # the one reported in the text

# ─── Long-time control run (invoke with `--longtime`) ─────────────────
# Same h and dt as above; only the domain length and the final time
# change, so the two runs differ in one variable only.  The front sits
# near x = 0.15 + c* t, so t = 250 needs x ~ 36.4 plus a margin of many
# wing widths before the Neumann boundary.
LX_LONG, NX_LONG = 42.0, 8401     # h = 0.005, unchanged
T_END_LONG = 250.0
WINDOWS_LONG = [(55, 75), (75, 100), (100, 130), (130, 170),
                (170, 210), (210, 250)]
LONGTIME_NPZ = "fisher_kpp_longtime.npz"

# ─── Leading-edge diagnostic (invoke with `--edge`) ───────────────────
# The speed excess is saturated to within 0.2% by t = 60, so a shorter
# domain and a moderate final time suffice to inspect the front's
# internal structure.
LX_EDGE, NX_EDGE = 12.0, 2401     # h = 0.005, unchanged
T_END_EDGE = 60.0
EDGE_NPZ = "fisher_kpp_edge.npz"

C_FRONT = "#D55E00"
C_THEORY = "#0072B2"


def front_position(T_mid, x, level=LEVEL):
    """Right-most crossing of the given level, by linear interpolation."""
    idx = np.where(T_mid >= level)[0]
    if len(idx) == 0 or idx[-1] + 1 >= len(x):
        return np.nan
    i = idx[-1]
    t0, t1 = T_mid[i], T_mid[i + 1]
    if t0 == t1:
        return x[i]
    return x[i] + (x[i + 1] - x[i]) * (level - t0) / (t1 - t0)


def _aggressive_parameters() -> Parameters:
    return Parameters(alpha=AGGRESSIVE.alpha, beta=AGGRESSIVE.beta,
                      gamma=AGGRESSIVE.gamma, delta=AGGRESSIVE.delta,
                      omega=AGGRESSIVE.omega, delta1=AGGRESSIVE.delta1,
                      delta2=AGGRESSIVE.delta2)


def long_time_run() -> None:
    """Integrate to t = 250 on a longer domain and report the late-window
    speed ratios.

    Purpose.  On t in [55,75] the reference run above measures
    c_num/c* = 1.004, i.e. a 0.4% EXCESS over the linear KPP speed.  Two
    mechanisms could produce it, and they predict opposite trends:

      * a slowly decaying transient of the coupled interior zone, which
        would make the ratio fall back towards 1 as t grows;
      * a genuinely (weakly) pushed front -- the acid diffuses ahead of
        the leading edge and depresses N there, so the effective growth
        rate beta(1 - alpha N) exceeds the nominal beta(1 - alpha) that
        defines c* -- which would make the ratio settle ABOVE 1, and in
        fact rise slightly further as the Bramson deficit
        3/(2 lambda* t) dies off.

    The trajectory is written to `fisher_kpp_longtime.npz` so the window
    analysis can be redone without repeating the integration (~80 min).
    """
    p = _aggressive_parameters()
    r = p.beta * (1.0 - p.alpha)
    D = p.delta2
    c_star = 2.0 * np.sqrt(D * r)
    lam = np.sqrt(r / D)
    h = LX_LONG / (NX_LONG - 1)

    print(f"long-time control run: Lx = {LX_LONG}, Nx = {NX_LONG}, "
          f"h = {h:.5f}, dt = {DT}, T_end = {T_END_LONG}")
    print(f"c* = {c_star:.6f}, lambda* = {lam:.4f}, "
          f"expected x_front(T_end) ~ {X_SEED + c_star * T_END_LONG:.2f}")

    grid = Grid2D(Nx=NX_LONG, Ny=NY, Lx=LX_LONG, Ly=LY)
    solver = Solver(grid, p, dt=DT, enforce_clip=False)

    T = np.where(grid.X <= X_SEED, 1.0, 0.0)
    N = 1.0 - T
    H = np.zeros_like(T)

    times, fronts = [], []
    nsteps = int(T_END_LONG / DT)
    sample = int(0.2 / DT)
    for n in range(nsteps + 1):
        t = n * DT
        if n % sample == 0:
            times.append(t)
            fronts.append(front_position(T[NY // 2, :], grid.x))
        if n == nsteps:
            break
        N, T, H = solver.step(N, T, H)

    times = np.array(times)
    fronts = np.array(fronts)

    viol = solver.violation_report()
    worst = max(v for k, v in viol.items() if k != "n_steps")

    centres, speeds = [], []
    for a, b in WINDOWS_LONG:
        m = (times >= a) & (times <= b) & np.isfinite(fronts)
        slope = np.polyfit(times[m], fronts[m], 1)[0]
        centres.append(0.5 * (a + b))
        speeds.append(slope)
        # pulled-front prediction on the same window, Bramson included:
        # mean of c* - 3/(2 lambda* t) over [a,b]
        bram = c_star - (3.0 / (2.0 * lam)) * np.log(b / a) / (b - a)
        print(f"  t in [{a:3d},{b:3d}]: c_num = {slope:.6f}, "
              f"c_num/c* = {slope / c_star:.4f}   "
              f"(pulled+Bramson would give {bram / c_star:.4f})")

    print(f"  final front position   = {fronts[np.isfinite(fronts)][-1]:.3f} "
          f"(domain ends at {LX_LONG})")
    print(f"  worst box violation    = {worst:.2e}")

    np.savez(LONGTIME_NPZ, times=times, fronts=fronts,
             centres=np.array(centres), speeds=np.array(speeds),
             c_star=c_star, lam=lam, h=h, dt=DT, Lx=LX_LONG, Nx=NX_LONG,
             windows=np.array(WINDOWS_LONG), worst_violation=worst)
    print(f"wrote {LONGTIME_NPZ}")


def leading_edge_diagnostic() -> None:
    """Measure the depression of N just ahead of the leading edge.

    The long-time run shows the front travelling 0.53% faster than the
    linear KPP speed c* = 2 sqrt(delta2 beta (1-alpha)), which is built
    on the growth rate beta(1-alpha) frozen at N = 1.  The proposed
    mechanism is that acid produced inside the front diffuses a short
    way ahead of it and depresses N there, so that the growth rate the
    leading edge actually experiences, beta(1 - alpha N), is larger.

    This routine integrates to a moderate late time (the excess is
    already saturated to within 0.2% by t = 60) and reports N and H at
    the leading edge, together with the growth-rate excess they imply.
    """
    p = _aggressive_parameters()
    r = p.beta * (1.0 - p.alpha)
    c_star = 2.0 * np.sqrt(p.delta2 * r)

    grid = Grid2D(Nx=NX_EDGE, Ny=NY, Lx=LX_EDGE, Ly=LY)
    solver = Solver(grid, p, dt=DT, enforce_clip=False)
    T = np.where(grid.X <= X_SEED, 1.0, 0.0)
    N = 1.0 - T
    H = np.zeros_like(T)

    nsteps = int(T_END_EDGE / DT)
    for _ in range(nsteps):
        N, T, H = solver.step(N, T, H)

    j = NY // 2
    Tp, Np, Hp, x = T[j, :], N[j, :], H[j, :], grid.x
    x_half = front_position(Tp, x)
    print(f"leading-edge diagnostic at t = {T_END_EDGE} "
          f"(Lx = {LX_EDGE}, h = {LX_EDGE/(NX_EDGE-1):.5f})")
    print(f"  front (T = 0.5) at x = {x_half:.4f}")

    # N, H and the local growth-rate enhancement, sampled on successive
    # decades of T going out through the wing towards the leading edge
    print("      T level        x     x-x_half        N          H     "
          "  beta(1-aN)/beta(1-a)-1")
    for lev in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10):
        i = np.where(Tp >= lev)[0]
        if len(i) == 0:
            continue
        k = i[-1]
        exc = (1 - p.alpha * Np[k]) / (1 - p.alpha) - 1.0
        print(f"  {lev:9.0e}  {x[k]:8.4f}  {x[k]-x_half:+9.3f}  "
              f"{Np[k]:9.6f}  {Hp[k]:9.3e}  {exc*100:+9.3f}%")

    print("  The enhancement decays to zero as T -> 0, so the LINEAR")
    print("  spreading speed at the true leading edge is exactly c*;")
    print("  the excess speed comes from the front's interior, which is")
    print("  what makes the front pushed rather than pulled.")
    print(f"  c* = {c_star:.6f}")
    np.savez(EDGE_NPZ, x=x, N=Np, T=Tp, H=Hp, x_half=x_half,
             c_star=c_star, t=T_END_EDGE, h=LX_EDGE / (NX_EDGE - 1))
    print(f"wrote {EDGE_NPZ}")


def main() -> None:
    p = Parameters(alpha=AGGRESSIVE.alpha, beta=AGGRESSIVE.beta,
                   gamma=AGGRESSIVE.gamma, delta=AGGRESSIVE.delta,
                   omega=AGGRESSIVE.omega, delta1=AGGRESSIVE.delta1,
                   delta2=AGGRESSIVE.delta2)
    r = p.beta * (1.0 - p.alpha)          # linear growth rate at T = 0, N = 1
    D = p.delta2
    c_star = 2.0 * np.sqrt(D * r)
    lam = np.sqrt(r / D)                  # decay rate of the front wing
    width = np.sqrt(D / p.beta)
    h = LX / (NX - 1)

    print(f"c*            = 2 sqrt(D r) = {c_star:.5f}")
    print(f"wing width    = sqrt(D/beta) = {width:.5f}")
    print(f"mesh step h   = {h:.5f}  ->  {width / h:.1f} nodes across the wing")
    print(f"Bramson coeff = 3/(2 lambda*) = {3 / (2 * lam):.5f}")

    grid = Grid2D(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    solver = Solver(grid, p, dt=DT, enforce_clip=False)

    T = np.where(grid.X <= X_SEED, 1.0, 0.0)
    N = 1.0 - T
    H = np.zeros_like(T)

    snap_times = (5.0, 20.0, 40.0, 60.0, 75.0)
    snaps, times, fronts = [], [], []
    nsteps = int(T_END / DT)
    sample = int(0.2 / DT)
    for n in range(nsteps + 1):
        t = n * DT
        if n % sample == 0:
            times.append(t)
            fronts.append(front_position(T[NY // 2, :], grid.x))
        for ts in snap_times:
            if abs(t - ts) < 0.5 * DT:
                snaps.append((ts, T[NY // 2, :].copy()))
        if n == nsteps:
            break
        N, T, H = solver.step(N, T, H)

    times = np.array(times)
    fronts = np.array(fronts)

    # apparent speed on each window
    centres, speeds = [], []
    for a, b in WINDOWS:
        m = (times >= a) & (times <= b) & np.isfinite(fronts)
        slope = np.polyfit(times[m], fronts[m], 1)[0]
        centres.append(0.5 * (a + b))
        speeds.append(slope)
        print(f"  t in [{a:3d},{b:3d}]: c_num = {slope:.5f}, "
              f"c_num/c* = {slope / c_star:.4f}")
    a, b = FIT_WINDOW
    m = (times >= a) & (times <= b) & np.isfinite(fronts)
    slope, intercept = np.polyfit(times[m], fronts[m], 1)

    viol = solver.violation_report()
    worst = max(v for k, v in viol.items() if k != "n_steps")
    print(f"  worst box violation = {worst:.2e}")

    # ─── figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.7))

    ax = axes[0]
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(snaps)))
    for col, (ts, prof) in zip(cols, snaps):
        ax.plot(grid.x, prof, color=col, lw=2.0, label=rf"$t = {ts:.0f}$")
    ax.axhline(LEVEL, color="grey", ls=(0, (1, 2)), lw=1.1)
    ax.text(LX * 0.72, LEVEL + 0.03, r"level set $T = 0.5$", fontsize=9,
            color="grey")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$T(x,\, L_y/2,\, t)$")
    ax.set_title("(a) tumour profile, planar front", loc="left")
    ax.set_xlim(0, LX)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(times, fronts, color=C_FRONT, lw=2.0,
            label=r"measured $x_{\mathrm{front}}(t)$")
    tt = np.array([a, b])
    ax.plot(tt, slope * tt + intercept, color="k", ls=(0, (5, 2)), lw=1.6,
            label=rf"fit on $t \in [{a},{b}]$: slope $= {slope:.4f}$")
    ax.plot(tt, c_star * (tt - tt[0]) + (slope * tt[0] + intercept),
            color=C_THEORY, ls=(0, (1, 1.6)), lw=1.8,
            label=rf"KPP speed $c^{{*}} = {c_star:.4f}$")
    ax.axvspan(a, b, color="k", alpha=0.05)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"front position $x_{\mathrm{front}}(t)$")
    ax.set_title("(b) front trajectory", loc="left")
    ax.set_xlim(0, T_END)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[2]
    ax.axhline(1.0, color=C_THEORY, lw=1.8, ls=(0, (1, 1.6)),
               label=r"KPP minimal speed $c^{*}$")
    ax.plot(centres, np.array(speeds) / c_star, "o-", color=C_FRONT, lw=2.0,
            ms=8, label=r"measured $c_{\mathrm{num}} / c^{*}$")
    ax.annotate("old fit window\n(transient)", xy=(8, 0.93),
                xytext=(20, 0.955), fontsize=8.5, color="#666", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666"))
    ax.plot([8], [0.93], "x", ms=9, mew=2, color="#666")
    ax.set_xlabel(r"centre of the fitting window")
    ax.set_ylabel(r"$c_{\mathrm{num}} / c^{*}$")
    ax.set_title("(c) approach to the KPP speed", loc="left")
    ax.set_ylim(0.90, 1.03)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    for ax in axes:
        ax.grid(alpha=0.2, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("fisher_kpp_speed.pdf", bbox_inches="tight")
    print("wrote fisher_kpp_speed.pdf")


if __name__ == "__main__":
    import sys

    if "--longtime" in sys.argv:
        long_time_run()
    elif "--edge" in sys.argv:
        leading_edge_diagnostic()
    else:
        main()
