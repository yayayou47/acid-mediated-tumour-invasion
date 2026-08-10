"""
bistability.py -- The bistable regime (alpha > 1, delta > delta_c), where the
coexistence equilibrium E3 is a SADDLE of the kinetics, and the mechanism that
sets its spatial threshold.

Two runs are performed with IDENTICAL parameters, differing only in the initial
tumour burden:

    * a small seed  -> the tissue clears the tumour, u -> E1 = (1, 0, 0)
    * a large seed  -> the tumour invades,          u -> E2 = (0, 1, g/w)

which is the operational signature of bistability: the outcome is decided by
the initial condition and not by the parameters.  The runs use centred data,
which stay in the reflection-symmetric subspace; there E3 keeps a
one-dimensional unstable manifold and W^s(E3) separates the two basins.  For
general data on this domain it does not, and the departure from symmetry is
measured here at every recorded step so that the claim rests on a number.

Spatial means alone would be a poor defence of a statement about spatial
structure, so the figure also shows the field with the tissue boundary
{N = 0.5}, the radial acid profile
against the quasi-static Bessel solution that predicts the threshold, the
domain dependence of the critical radius, and its dependence on seed density.
The last two panels read the results of `critical_radius.py` from
critical_radius_results.json rather than recomputing them.

Produces `bistability.pdf`.

Author: Yaya Youssouf Yaya
Date:   2026-08-04
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0, i1, k0, k1

from critical_radius import bessel_prediction
from main import Grid2D, Parameters, Solver
from params import SEED

# Embed TrueType rather than Type 3 fonts: Type 3 is rejected by most
# publisher production pipelines and makes the text non-extractable.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
np.random.seed(SEED)

# ─── Bistable parameter set: alpha > 1 AND delta > delta_c ───────────
ALPHA, BETA, GAMMA, OMEGA, DELTA = 1.2, 1.5, 5.0, 8.0, 2.5
DELTA1 = DELTA2 = 5e-3
DELTA_C = OMEGA / GAMMA                      # 1.6  ->  delta/delta_c = 1.5625
ELL_H = OMEGA ** -0.5                        # acid screening length

NX = NY = 101
LX = LY = 2.0
DT = 5e-3
T_FINAL = 40.0
T_SNAP = 3.0            # field snapshot: the tissue boundary is still moving
T_LATE = 6.0            # by then the whole square is receding; reported, not drawn
T_PROFILE = 0.5         # acid profile: four relaxation times 1/omega, and the
                        # tumour has barely moved
SEED_AMP = 0.95

RESULTS_FILE = "critical_radius_results.json"

C_N = "#0072B2"     # normal cells
C_T = "#D55E00"     # tumour cells
C_H = "#009E73"     # acid


def initial_data(grid, radius: float):
    """Healthy tissue at carrying capacity, circular tumour seed at centre."""
    X, Y = grid.X, grid.Y
    disc = np.sqrt((X - LX / 2) ** 2 + (Y - LY / 2) ** 2) <= radius
    T0 = SEED_AMP * disc.astype(float)
    return 1.0 - T0, T0, np.zeros_like(T0)


def reflection_asymmetry(*fields) -> float:
    """Largest departure from the symmetry group of the centred initial data.

    The modes (1,0), (0,1) and (1,1), which are the ones that make E3 unstable
    in four directions on this domain, are odd under those reflections and so
    are absent from the subspace these runs live in.  The manuscript quotes the
    value returned here, and it should stay at the level of round-off.
    """
    worst = 0.0
    for f in fields:
        worst = max(worst,
                    float(np.abs(f - f[::-1, :]).max()),
                    float(np.abs(f - f[:, ::-1]).max()),
                    float(np.abs(f - f.T).max()))
    return worst


def radial_average(grid, field, nbins=60):
    """Azimuthal average of a field about the centre of the square."""
    r = np.sqrt((grid.X - LX / 2) ** 2 + (grid.Y - LY / 2) ** 2)
    edges = np.linspace(0.0, 0.5 * LX, nbins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nbins - 1)
    total = np.bincount(idx, weights=field.ravel(), minlength=nbins)
    count = np.bincount(idx, minlength=nbins)
    return 0.5 * (edges[:-1] + edges[1:]), total / np.maximum(count, 1)


def bessel_profile(r, radius, amp=SEED_AMP):
    """Quasi-static acid profile of a disc source in an unbounded medium.

    H solves -Delta H + omega H = gamma*amp on the disc of the given radius and
    -Delta H + omega H = 0 outside, with H and its derivative continuous.  At
    the centre this reduces to equation (bessel) of the manuscript.
    """
    s = np.sqrt(OMEGA)
    inside = (GAMMA * amp / OMEGA) * (1.0 - s * radius * k1(s * radius)
                                      * i0(s * np.asarray(r)))
    outside = (GAMMA * amp / OMEGA) * s * radius * i1(s * radius) \
        * k0(s * np.maximum(np.asarray(r), 1e-12))
    return np.where(np.asarray(r) <= radius, inside, outside)


def run(radius: float):
    """One run, returning the means, a snapshot, and the early acid profile."""
    grid = Grid2D(Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    p = Parameters(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                   omega=OMEGA, delta1=DELTA1, delta2=DELTA2)
    solver = Solver(grid, p, dt=DT, enforce_clip=False)
    N, T, H = initial_data(grid, radius)

    nsteps = int(T_FINAL / DT)
    ts, mN, mT = [], [], []
    asym, snap, profile, kill = 0.0, None, None, None
    receding = {}
    ic, jc = grid.Ny // 2, grid.Nx // 2
    for n in range(nsteps + 1):
        t = n * DT
        if n % 20 == 0:
            ts.append(t)
            mN.append(float(N.mean()))
            mT.append(float(T.mean()))
            asym = max(asym, reflection_asymmetry(N, T, H))
        if kill is None and DELTA * H[ic, jc] > 1.0:
            # The instant the centre becomes lethal to normal cells, and the
            # tumour density there at that instant.  The quasi-static estimate
            # assumes the seed still carries its initial density; this is what
            # it actually carries by then.
            kill = (t, float(T[ic, jc]), float(N[ic, jc]))
        if abs(t - T_SNAP) < 0.5 * DT:
            snap = (N.copy(), T.copy(), H.copy())
        for t_mark in (T_SNAP, T_LATE):
            if abs(t - t_mark) < 0.5 * DT:
                receding[t_mark] = float((N < 0.5).mean())
        if abs(t - T_PROFILE) < 0.5 * DT:
            profile = radial_average(grid, H)
        if n == nsteps:
            break
        N, T, H = solver.step(N, T, H)
    return {"t": np.array(ts), "N": np.array(mN), "T": np.array(mT),
            "snap": snap, "profile": profile, "grid": grid, "kill": kill,
            "receding": receding,
            "viol": solver.violation_report(), "asym": asym, "radius": radius}


def load_results():
    """Read the bisections of critical_radius.py, grouped by family."""
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit(
            f"{RESULTS_FILE} not found: run `python3 critical_radius.py --all` "
            "first, since panels (e) and (f) report its measurements.")
    with open(RESULTS_FILE) as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        if "R_c" in r:
            out.setdefault(r["family"], []).append(r)
    return out


# ══════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════

def panel_means(ax, res, title, e3, show_legend=False):
    ax.plot(res["t"], res["N"], color=C_N, lw=2.2, label=r"$\langle N \rangle$")
    ax.plot(res["t"], res["T"], color=C_T, lw=2.2, label=r"$\langle T \rangle$")
    ax.axhline(1.0, color="k", lw=0.9, ls=(0, (1, 2)), alpha=0.6)
    ax.axhline(0.0, color="k", lw=0.9, ls=(0, (1, 2)), alpha=0.6)
    if e3 is not None:
        ax.axhline(e3[0], color=C_N, lw=1.0, ls=(0, (5, 2)), alpha=0.7)
        ax.axhline(e3[1], color=C_T, lw=1.0, ls=(0, (5, 2)), alpha=0.7)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(r"time $t$")
    ax.set_xlim(0, T_FINAL)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="center right", frameon=False, fontsize=10)


def panel_field(ax, res):
    N, T, H = res["snap"]
    grid = res["grid"]
    im = ax.imshow(T, origin="lower", extent=grid.extent, cmap="magma",
                   vmin=0.0, vmax=1.0)
    # The tissue boundary, as in the regime figure.  We do not draw the set
    # where the tissue is losing ground, {delta*H > 1 - N}: at carrying
    # capacity 1 - N vanishes, so that set includes the whole far field
    # wherever any acid has diffused, and a contour of it would be read as
    # the destruction front, which it is not.
    ax.contour(grid.X, grid.Y, N, levels=[0.5], colors="#00E5FF",
               linewidths=1.8)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(LX / 2 + res["radius"] * np.cos(theta),
            LY / 2 + res["radius"] * np.sin(theta),
            color="w", lw=1.0, ls=(0, (4, 3)))
    ax.set_title(rf"(c) $T$ at $t = {T_SNAP:g}$, seed radius "
                 rf"${res['radius']}$", loc="left", fontsize=11)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return im


def panel_profile(ax, small, large):
    for res, colour, style in ((small, C_N, "-"), (large, C_T, "-")):
        r, h = res["profile"]
        ax.plot(r, h, style, color=colour, lw=2.0,
                label=rf"measured, $R = {res['radius']}$")
        ax.plot(r, bessel_profile(r, res["radius"]), ls=(0, (5, 2)),
                color=colour, lw=1.3, alpha=0.85,
                label=rf"quasi-static, $R = {res['radius']}$")
    ax.axhline(1.0 / DELTA, color="k", lw=1.0, ls=(0, (1, 2)))
    ax.annotate(r"$\delta H = 1$", xy=(0.62, 1.0 / DELTA), fontsize=9,
                va="bottom")
    ax.axvline(ELL_H, color="#666", lw=0.9)
    ax.annotate(r"$\ell_H$", xy=(ELL_H, 0.02), fontsize=9, color="#666",
                xytext=(ELL_H + 0.02, 0.02))
    ax.set_title(rf"(d) acid profile at $t = {T_PROFILE:g}$",
                 loc="left", fontsize=11)
    ax.set_xlabel(r"distance $r$ from the seed centre")
    ax.set_ylabel(r"$H$")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1.28 * float(max(h.max() for _, h in
                                    (small["profile"], large["profile"]))))
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2,
              columnspacing=0.8, handlelength=1.4)
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)


def panel_domains(ax, rows, R_qs):
    rows = sorted(rows, key=lambda r: r["L"])
    L = np.array([r["L"] for r in rows])
    Rc = np.array([r["R_c"] for r in rows])
    err = np.array([r["half_width"] for r in rows])
    ax.errorbar(L, Rc, yerr=err, marker="o", ms=5, lw=1.8, color=C_T,
                capsize=3, label=r"measured $R_c$")
    ax.axhline(R_qs, color="#444", lw=1.2, ls=(0, (5, 2)),
               label=r"unbounded medium, $1.81\,\ell_H$")
    ax.set_title("(e) domain dependence", loc="left", fontsize=11)
    ax.set_xlabel(r"box side $L$")
    ax.set_ylabel(r"critical radius $R_c$")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)


def panel_density(ax, amps, radii, errs):
    ax.errorbar(amps, radii, yerr=errs, marker="s", ms=5, lw=1.8, color=C_N,
                capsize=3)
    ax.fill_between(amps, radii, 1.45, color=C_T, alpha=0.12)
    ax.fill_between(amps, 0.0, radii, color=C_N, alpha=0.10)
    ax.annotate("invasion", xy=(0.80, 1.15), fontsize=10, color=C_T)
    ax.annotate("clearance", xy=(0.80, 0.20), fontsize=10, color=C_N)
    ax.set_xlim(0.42, 1.02)
    ax.set_title("(f) threshold in the size--density plane", loc="left",
                 fontsize=11)
    ax.set_xlabel(r"seed density $T_0$")
    ax.set_ylabel(r"critical radius $R_c$")
    ax.set_ylim(0.0, 1.45)
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    p_probe = Parameters(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA,
                         omega=OMEGA, delta1=DELTA1, delta2=DELTA2)
    e3 = p_probe.E3
    print(f"regime = {p_probe.regime}, delta/delta_c = {DELTA/DELTA_C:.4f}")
    print(f"E3 = {e3},  saddle = {p_probe.E3_is_saddle}")
    print(f"ell_H = {ELL_H:.4f}")

    families = load_results()
    domains = families.get("domains", [])
    R_qs = bessel_prediction()

    small = run(radius=0.35)     # sub-threshold seed
    large = run(radius=0.65)     # supra-threshold seed

    # density panel: the three amplitudes measured at L = 2
    amp_rows = {0.95: next(r for r in domains if abs(r["L"] - 2.0) < 1e-9)}
    for r in families.get("amplitude", []):
        amp_rows[float(r["amp"])] = r
    amps = sorted(amp_rows)
    radii = [amp_rows[a]["R_c"] for a in amps]
    errs = [amp_rows[a]["half_width"] for a in amps]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    panel_means(axes[0, 0], small,
                r"(a) seed radius $0.35$ $\rightarrow$ cleared ($E_1$)", e3,
                show_legend=True)
    panel_means(axes[0, 1], large,
                r"(b) seed radius $0.65$ $\rightarrow$ invasion ($E_2$)", e3)
    axes[0, 0].set_ylabel("spatial mean density")
    im = panel_field(axes[0, 2], large)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.03)
    panel_profile(axes[1, 0], small, large)
    panel_domains(axes[1, 1], domains, R_qs)
    panel_density(axes[1, 2], amps, radii, errs)

    for ax in axes.ravel():
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        rf"Bistable regime  $\alpha = {ALPHA} > 1$, "
        rf"$\delta/\delta_c = {DELTA/DELTA_C:.2f} > 1$: "
        r"identical parameters, opposite outcomes, and the length that "
        r"separates them",
        fontsize=12.5, y=0.995)
    fig.tight_layout()
    fig.savefig("bistability.pdf", bbox_inches="tight")
    print("wrote bistability.pdf")

    for name, res in (("small", small), ("large", large)):
        worst = max(v for k, v in res["viol"].items() if k != "n_steps")
        print(f"  {name} seed: final <N> = {res['N'][-1]:.4f}, "
              f"<T> = {res['T'][-1]:.4f}, worst box violation = {worst:.2e}, "
              f"reflection asymmetry = {res['asym']:.2e}")
        for t_mark, frac in sorted(res["receding"].items()):
            print(f"    at t = {t_mark:g}, the tissue is below half its "
                  f"carrying capacity on {100 * frac:.0f} % of the square")
        if res["kill"] is None:
            print("    the centre never becomes lethal to normal cells")
        else:
            t, Tc, Nc = res["kill"]
            print(f"    delta*H > 1 at the centre from t = {t:.2f}, when the "
                  f"tumour density there is {Tc:.3f} (initially {SEED_AMP}) "
                  f"and the tissue {Nc:.3f}")
    r, h = large["profile"]
    pred = bessel_profile(r, large["radius"])
    inside = r <= large["radius"]
    print(f"  acid profile at t = {T_PROFILE}: max relative departure from the "
          f"quasi-static solution inside the seed = "
          f"{np.max(np.abs(h[inside] - pred[inside]) / pred[inside]):.3f}")
    print(f"  R_c(L=2) = {amp_rows[0.95]['R_c']:.4f}, "
          f"unbounded prediction {R_qs:.4f} = {R_qs / ELL_H:.2f} ell_H")


if __name__ == "__main__":
    main()
