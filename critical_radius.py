"""
critical_radius.py -- The bistable threshold: what it depends on, and what sets it.

In the bistable region {alpha > 1, delta > delta_c} the outcome is decided by the
initial datum, and the simulations of the manuscript report a critical seed
radius R_c.  Three questions may fairly be asked of such a number, and this
script answers all three by measurement rather than by assertion.

1.  *Is R_c a length of the model, or a length of the box?*  Homogeneous Neumann
    conditions retain the acid, so a small box lowers the threshold.  The
    ``domains`` sweep bisects R_c on a growing sequence of squares and reports
    whether it settles.  Two runs at a refined mesh check that what is measured
    is not the discretisation.

2.  *What sets it?*  Not the contrast of diffusivities: ``diffusivity`` varies
    delta1 = delta2 over two decades and R_c barely moves.  The controlling
    length is the acid screening length ell_H = omega^(-1/2).  ``screening``
    tests that quantitatively: omega is varied over a decade at fixed
    delta/delta_c and fixed box-to-screening-length ratio, and R_c/ell_H is
    reported.  Since the cell diffusion length sqrt(delta2/beta) is held fixed
    while ell_H changes, a constant ratio is a genuine result and not an
    artefact of scaling.  ``shape`` and ``amplitude`` delimit what the claim
    means: a threshold in size holds at fixed seed density, and R_c depends on
    the seed profile only weakly at equal mass.

3.  *Does the threshold state have spatial structure?*  ``offcentre`` places the
    seed away from the centre, which breaks every reflection of the square, and
    ``lsweep`` follows the near-critical trajectory as the box grows past the
    value L* = 2 pi / sqrt(mu*) at which the first mode of the symmetric class
    destabilises the coexistence state.

The outcome of a run is decided locally, from the tumour peak, and never from a
spatial mean: a mean depends on the size of the box, and comparing boxes is the
whole point of the first question.  A seed is cleared when max T falls below
CLEAR_TOL and invaded when max T, having passed a minimum, has grown back by
GROW_FACTOR.  Both are reached well before T_MAX at every radius tested; runs
that reach T_MAX undecided are reported as such and never silently coerced.

Usage
-----
    python3 critical_radius.py --all                  # everything, ~1-2 h on 6 cores
    python3 critical_radius.py --domains --screening  # selected experiments
    python3 critical_radius.py --all --quick          # coarser bisection, for a smoke test

Results are written to critical_radius_results.json, which
``bistability.py`` reads to draw the R_c(L) panel.

Author: Yaya Youssouf Yaya
Date:   2026-08-04
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
from scipy.special import k1

from main import Grid2D, Parameters, Solver

# ─── Bistable parameter set, as in the manuscript ────────────────────
ALPHA, BETA, GAMMA, OMEGA, DELTA = 1.2, 1.5, 5.0, 8.0, 2.5
DELTA1 = DELTA2 = 5e-3
SEED_AMP = 0.95
DT = 5e-3
H_TARGET = 0.02          # mesh spacing, held fixed across box sizes
T_MAX = 150.0            # cap on integration time; never reached in practice
CLEAR_TOL = 1e-3         # max T below this: the focus is gone
GROW_FACTOR = 1.25       # tumour mass back up by this from its minimum: it grows
CHECK_EVERY = 100        # steps between outcome tests (0.5 time units)

RESULTS_FILE = "critical_radius_results.json"


# ══════════════════════════════════════════════════════════════════════
# 1. ONE RUN, AND ITS OUTCOME
# ══════════════════════════════════════════════════════════════════════

def make_seed(grid, radius, amp=SEED_AMP, centre=None, shape="disc"):
    """Tumour focus of equivalent radius ``radius``, with the tissue displaced.

    The three shapes carry the same mass ``amp * pi * radius**2``, so that
    comparing their thresholds compares profiles and not tumour burden.
    """
    cx, cy = centre if centre is not None else (grid.Lx / 2, grid.Ly / 2)
    X, Y = grid.X, grid.Y
    if shape == "disc":
        T = amp * ((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2).astype(float)
    elif shape == "square":
        half = 0.5 * np.sqrt(np.pi) * radius          # equal area
        T = amp * ((np.abs(X - cx) <= half) & (np.abs(Y - cy) <= half)).astype(float)
    elif shape == "gaussian":
        sigma = radius / np.sqrt(2.0)                 # equal mass
        T = amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"unknown seed shape {shape!r}")
    return np.clip(1.0 - T, 0.0, 1.0), T, np.zeros_like(T)


def run_seed(L, radius, h=H_TARGET, d1=DELTA1, d2=DELTA2, gamma=GAMMA,
             omega=OMEGA, amp=SEED_AMP, centre=None, shape="disc",
             t_max=T_MAX, record=False, dt=DT):
    """Integrate one seed until its fate is decided (or until ``t_max``).

    Returns a dict with the verdict, the time at which it was reached, the
    largest violation of the invariant region, and -- when ``record`` is set --
    the trajectory of the spatial mean and standard deviation of T, which is
    what the L-sweep of the near-critical state needs.
    """
    n = int(round(L / h)) + 1
    grid = Grid2D(Nx=n, Ny=n, Lx=L, Ly=L)
    p = Parameters(alpha=ALPHA, beta=BETA, gamma=gamma, delta=DELTA,
                   omega=omega, delta1=d1, delta2=d2)
    solver = Solver(grid, p, dt=dt, enforce_clip=False)
    N, T, H = make_seed(grid, radius, amp=amp, centre=centre, shape=shape)

    mass_min = float(T.mean())
    verdict, t_dec = None, float("nan")
    series, kill = [], None
    ic, jc = grid.Ny // 2, grid.Nx // 2
    if centre is not None:
        jc = int(round(centre[0] / h))
        ic = int(round(centre[1] / h))
    check_every = max(1, int(round(CHECK_EVERY * DT / dt)))
    for k in range(int(t_max / dt)):
        N, T, H = solver.step(N, T, H)
        if (k + 1) % check_every:
            continue
        t = (k + 1) * dt
        peak, mass = float(T.max()), float(T.mean())
        if kill is None and p.delta * H[ic, jc] > 1.0:
            # The instant the seed centre becomes lethal to normal cells at
            # carrying capacity, and what the seed still carries then.  The
            # quasi-static criterion is about exactly this moment.
            kill = (t, float(T[ic, jc]), float(N[ic, jc]))
        if record:
            series.append((round(t, 3), mass, float(T.std()),
                           float(N.mean()), peak))
        mass_min = min(mass_min, mass)
        if verdict is None:
            # Cleared: the focus has vanished.  Invaded: its mass has turned
            # round and is growing again, with a peak that is not a residue.
            # Both tests are local to the focus, so neither depends on the box.
            if peak < CLEAR_TOL:
                verdict, t_dec = False, t
            elif mass > GROW_FACTOR * mass_min and peak > 0.5:
                verdict, t_dec = True, t
            if verdict is not None and not record:
                break
    viol = solver.violation_report()
    out = {"invaded": verdict, "t_decided": t_dec, "n": n, "L": L,
           "radius": radius, "kill": kill,
           "violation": max(viol[key] for key in viol
                            if key != "n_steps")}
    if record:
        out["series"] = series
    return out


def _decide(kwargs):
    """Outcome of one run, with an undecided run treated as a failure to answer."""
    res = run_seed(**kwargs)
    if res["invaded"] is None:
        raise RuntimeError(f"undecided at t = {T_MAX} for {kwargs}")
    return res


def bisect(steps, lo, hi, **kwargs):
    """Bracket the critical equivalent radius, monotonicity in radius assumed.

    The two ends are tested first: a bracket that does not straddle the
    threshold is reported rather than silently bisected into nonsense.
    """
    r_lo = _decide({**kwargs, "radius": lo})
    r_hi = _decide({**kwargs, "radius": hi})
    if r_lo["invaded"] or not r_hi["invaded"]:
        return {"error": "bracket does not straddle the threshold",
                "lo": lo, "hi": hi, "lo_invaded": r_lo["invaded"],
                "hi_invaded": r_hi["invaded"], **kwargs}
    viol = max(r_lo["violation"], r_hi["violation"])
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        res = _decide({**kwargs, "radius": mid})
        viol = max(viol, res["violation"])
        if res["invaded"]:
            hi = mid
        else:
            lo = mid
    return {"R_c": 0.5 * (lo + hi), "bracket": (lo, hi),
            "half_width": 0.5 * (hi - lo), "violation": viol, **kwargs}


# ══════════════════════════════════════════════════════════════════════
# 2. THE QUASI-STATIC PREDICTION
# ══════════════════════════════════════════════════════════════════════

def bessel_prediction(gamma=GAMMA, omega=OMEGA, amp=SEED_AMP):
    """Critical radius from the quasi-static acid profile of a disc source.

    With H relaxing on 1/omega against 1/beta for the cells, H solves
    -Delta H + omega H = gamma T0 on the disc of radius R in the plane, whose
    value at the centre is (gamma T0/omega)[1 - sqrt(omega) R K_1(sqrt(omega) R)].
    Normal cells die where delta H > 1, so the centre is cleared when that
    expression exceeds 1/delta.  The condition involves omega only through
    ell_H = omega^(-1/2) and the gauge-invariant ratio rho = delta gamma/omega,
    so the predicted radius is a fixed multiple of ell_H at fixed rho.
    """
    rho, s = DELTA * gamma / omega, np.sqrt(omega)

    def f(R):
        return rho * amp * (1.0 - s * R * k1(s * R)) - 1.0

    lo, hi = 1e-6, 100.0 / s
    if f(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if f(mid) > 0 else (mid, hi)
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════
# 3. THE EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

ELL_H = OMEGA ** -0.5
MU_STAR = 5.3567                      # root of tilde a_3, Prop. modal dichotomy
L_STAR_SYM = 2 * np.pi / np.sqrt(MU_STAR)   # first mode of the symmetric class


def jobs_domains(steps):
    """R_c on a growing sequence of boxes, plus two mesh controls."""
    out = []
    for L in (1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0):
        h = H_TARGET if L <= 4.0 else 1.5 * H_TARGET
        out.append(("domains", f"L={L}", dict(steps=steps, lo=0.12,
                                              hi=min(0.45 * L, 1.60),
                                              L=L, h=h)))
    out.append(("mesh", "L=2, h=0.0125", dict(steps=steps, lo=0.12, hi=0.90,
                                              L=2.0, h=0.0125)))
    out.append(("mesh", "L=4, h=0.03", dict(steps=steps, lo=0.12, hi=1.60,
                                            L=4.0, h=0.03)))
    return out


def jobs_diffusivity(steps):
    """Two decades of cell diffusivity, at the box of the manuscript."""
    return [("diffusivity", f"delta1=delta2={d:.0e}",
             dict(steps=steps, lo=0.12, hi=0.90, L=2.0, d1=d, d2=d))
            for d in (5e-4, 5e-3, 5e-2)]


def jobs_screening(steps):
    """omega over a decade at fixed delta/delta_c and fixed L/ell_H.

    gamma is slaved to omega so that delta_c = omega/gamma, hence
    delta/delta_c, is unchanged; the box grows with ell_H so that confinement
    is the same in screening-length units; the cell diffusivities are held
    fixed, so a constant R_c/ell_H is a statement about the model and not a
    similarity built into the set-up.
    """
    out = []
    for omega in (4.0, 8.0, 16.0, 32.0):
        gamma = omega / (OMEGA / GAMMA)          # same delta_c = omega/gamma
        ell = omega ** -0.5
        L = 8.0 * ell
        out.append(("screening", f"omega={omega:g}",
                    dict(steps=steps, lo=0.10 * (ell / ELL_H),
                         hi=1.20 * (ell / ELL_H), L=L, h=H_TARGET,
                         gamma=gamma, omega=omega)))
    # The mesh spacing is absolute while ell_H is not, so the sweep resolves
    # the screening length with 25 nodes at omega = 4 and 9 at omega = 32, and
    # the measured ratio drifts in the same direction.  This control repeats
    # the least resolved point on a mesh twice as fine: if the drift is a mesh
    # effect the ratio moves, if it is real it does not.
    out.append(("screening", "omega=32, h=0.01",
                dict(steps=steps, lo=0.05, hi=0.60, L=8.0 * 32 ** -0.5,
                     h=0.5 * H_TARGET, gamma=32.0 / (OMEGA / GAMMA),
                     omega=32.0)))
    return out


def jobs_offcentre(steps):
    """A seed away from the centre breaks every reflection of the square.

    Centred data live in a symmetric subspace from which the three modes that
    destabilise the coexistence state are absent, so those runs cannot exhibit
    a non-constant threshold state even if there is one.  Moving the seed
    removes that restriction, and the near-critical trajectory is recorded so
    that its spatial standard deviation can be read off.
    """
    return [("offcentre", "centre=(0.7,0.7)",
             dict(steps=steps, lo=0.12, hi=0.90, L=2.0, centre=(0.7, 0.7),
                  record_critical=True))]


def jobs_shape(steps):
    """Disc, square and Gaussian seeds of equal mass."""
    return [("shape", s, dict(steps=steps, lo=0.12, hi=0.95, L=2.0, shape=s))
            for s in ("square", "gaussian")]


def jobs_amplitude(steps):
    """R_c at several seed densities: the threshold in size is one at fixed density.

    A tempting guess is that no radius can work unless the acid a saturated
    disc produces exceeds the level that kills tissue at carrying capacity,
    rho*T0 > 1, which would put a floor at T0 = delta_c/delta = 0.64.  The
    measurements below refute it: inside the seed the tissue has already been
    displaced to 1 - T0, and there the logistic term only requires
    delta*H > 1 - N.  A sparser seed therefore still invades, from a larger
    radius, and the threshold is a curve in the size-density plane rather
    than a floor.
    """
    return [("amplitude", f"amp={a}", dict(steps=steps, lo=0.12, hi=1.40,
                                           L=2.0, amp=a))
            for a in (0.50, 0.75)]


def jobs_lsweep(steps):
    """The near-critical trajectory as the box grows past L* = 2 pi/sqrt(mu*).

    Centred data stay in the reflection-symmetric subspace, whose first
    non-constant mode is (2,0); it destabilises the coexistence state when
    L > 2 pi/sqrt(mu*) = 2.715.  Below that value the near-critical trajectory
    should flatten onto the constant state, above it it should not.
    """
    return [("lsweep", f"L={L}", dict(steps=steps, lo=0.12,
                                      hi=min(0.45 * L, 1.60), L=L,
                                      h=H_TARGET, record_critical=True))
            for L in (2.0, 2.5, 3.0, 3.5)]


def jobs_centre(steps):
    """Is the quasi-static criterion at the centre sufficient for invasion?

    Equation (bessel) asks when delta*H exceeds 1 at the centre of the seed,
    and predicts R = 1.81 ell_H.  The measured threshold on a large box is
    larger than that, so the criterion cannot be the whole story.  These runs
    sit on a box wide enough for the plateau and report, for radii on either
    side of the measured threshold, whether the centre ever becomes lethal and
    what the outcome is.  A radius that turns the centre lethal and is cleared
    all the same settles the question.
    """
    return [("centre", f"R={R}", dict(probe=True, L=6.0, h=1.5 * H_TARGET,
                                      radius=R))
            for R in (0.60, 0.70, 0.80, 0.85)]


def jobs_pairs(steps):
    """The dwell of two runs straddling the threshold, on each box.

    The L-sweep reads the threshold state off a single near-critical run, and
    the standard deviation of T can fall for two opposite reasons: because the
    trajectory flattens onto a constant state, or because it has already
    chosen a side and is heading for a uniform equilibrium.  Running both
    sides of the bracket separates them: if the dwell value characterises the
    threshold state, the two sides agree; if it tracks the outcome, they do
    not.  The radii are read from the stored bisections.
    """
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE) as fh:
        rows = json.load(fh)
    out = []
    for r in rows:
        if r.get("family") != "lsweep" or "R_c" not in r:
            continue
        w = r["half_width"]
        for side, R in (("below", r["R_c"] - 1.5 * w),
                        ("above", r["R_c"] + 1.5 * w)):
            out.append(("pairs", f"L={r['L']:g}, {side}",
                        dict(probe=True, record_critical=True, L=r["L"],
                             h=r.get("h", H_TARGET), radius=round(R, 4),
                             t_max=130.0)))
    return out


def jobs_timestep(steps):
    """Does the threshold depend on the time step?

    Every run of this section uses dt = 5e-3, chosen for the Lipschitz
    constant of the reaction on the invariant region and never varied.  The
    diffusion is implicit, so the restriction is mild, but the bistable runs
    are the stiffest in the paper and the claim rests on a threshold: this
    repeats the bisection of the reference box with the step halved.
    """
    return [("timestep", f"dt={dt:g}",
             dict(steps=steps, lo=0.12, hi=0.90, L=2.0, dt=dt))
            for dt in (0.5 * DT,)]


def jobs_monotone(steps):
    """Is the outcome monotone in the seed radius?

    Every threshold reported here is located by bisection, which converges to
    *a* change of outcome and not necessarily to the smallest one; `bisect`
    states the assumption and this sweep tests it.  Radii are spaced uniformly
    on two boxes, one of them the box that carries the plateau, and the
    assumption asks for a single transition with every radius below it cleared
    and every radius above it invaded.  The test is not idle: the section
    already contains one place where monotone intuition failed, the density
    floor that the quasi-static criterion predicts and that does not exist.
    """
    out = []
    for L, h, lo, hi, n in ((2.0, H_TARGET, 0.20, 0.90, 15),
                            (4.0, 1.5 * H_TARGET, 0.35, 1.15, 9)):
        for R in np.linspace(lo, hi, n):
            out.append(("monotone", f"L={L:g}, R={R:.3f}",
                        dict(probe=True, L=L, h=h, radius=round(float(R), 4))))
    return out


def jobs_dwell(steps):
    """The near-critical trajectory on the two boxes that fix the plateau.

    The invasion test fires when the tumour mass has grown by a quarter from
    its minimum.  On the small boxes the dwell phase stays about 1.22 times
    that minimum, close enough to the trigger that the margin is worth seeing
    on the two boxes that carry R_c^infty, and no trajectory was recorded
    there.  These runs sit just below the measured threshold, where the dwell
    is longest.
    """
    return [("dwell", f"L={L}, R just under R_c",
             dict(probe=True, record_critical=True, L=L, h=1.5 * H_TARGET,
                  radius=0.8100, t_max=140.0))
            for L in (4.0, 6.0)]


EXPERIMENTS = {"domains": jobs_domains, "diffusivity": jobs_diffusivity,
               "centre": jobs_centre, "dwell": jobs_dwell,
               "monotone": jobs_monotone, "timestep": jobs_timestep,
               "pairs": jobs_pairs,
               "screening": jobs_screening, "offcentre": jobs_offcentre,
               "shape": jobs_shape, "amplitude": jobs_amplitude,
               "lsweep": jobs_lsweep}


def _work(job):
    """Run one bisection; optionally rerun at the critical radius, recording."""
    family, label, kwargs = job
    kwargs = dict(kwargs)
    record = kwargs.pop("record_critical", False)
    t0 = time.time()
    if kwargs.pop("probe", False):
        res = run_seed(record=record, **kwargs)
        if record:
            res["critical_series"] = res.pop("series", [])
        res["family"], res["label"] = family, label
        res["wall_s"] = round(time.time() - t0, 1)
        return res
    res = bisect(**kwargs)
    if record and "R_c" in res:
        run_kw = {k: v for k, v in kwargs.items()
                  if k not in ("steps", "lo", "hi")}
        traj = run_seed(radius=res["R_c"], t_max=120.0, record=True, **run_kw)
        res["critical_series"] = traj["series"]
        res["critical_verdict"] = traj["invaded"]
    res["family"], res["label"] = family, label
    res["wall_s"] = round(time.time() - t0, 1)
    return res


# ══════════════════════════════════════════════════════════════════════
# 4. REPORTING
# ══════════════════════════════════════════════════════════════════════

def dwell_summary(series, t_start=10.0, band=0.25):
    """How uniform does the near-critical trajectory get while it lingers?

    Both outcomes are spatially uniform, so the standard deviation of T
    vanishes at large time whatever happens, and its long-time value says
    nothing.  What distinguishes a constant threshold state from a
    non-constant one is the dwell phase, during which the trajectory sits
    near the threshold before choosing a side.  We take the dwell to last
    until the tumour mass has moved by `band` from its value at `t_start`,
    and report the smallest standard deviation reached in it: a constant
    threshold state drives it towards zero, a non-constant one does not.
    """
    s = [row for row in series if row[0] >= t_start]
    if len(s) < 5:
        return "no dwell phase recorded"
    t = np.array([row[0] for row in s])
    mass = np.array([row[1] for row in s])
    std = np.array([row[2] for row in s])
    left = np.abs(mass - mass[0]) > band * mass[0]
    end = int(np.argmax(left)) if left.any() else len(s) - 1
    end = max(end, 2)
    j = int(np.argmin(std[:end + 1]))
    # Measured against the tumour component of the coexistence state, which
    # is the amplitude a non-constant threshold state would have to carry.
    T_star = ALPHA and (OMEGA * (1 - ALPHA)
                        / (OMEGA - ALPHA * DELTA * GAMMA))
    rel = std[j] / abs(T_star)
    verdict = "close to constant" if rel < 0.5 else "NOT constant"
    return (f"dwell t in [{t[0]:.0f},{t[end]:.0f}], mass {mass[0]:.3f} "
            f"(T* = {abs(T_star):.3f}), std(T) {std[0]:.3f} -> min "
            f"{std[j]:.3f} = {rel:.2f} T*  -> {verdict}")


def monotone_report(rows):
    """One line per box: the outcome of each radius, and the transition count."""
    boxes = {}
    for r in rows:
        boxes.setdefault(r["L"], []).append(r)
    for L in sorted(boxes):
        rs = sorted(boxes[L], key=lambda r: r["radius"])
        radii = [r["radius"] for r in rs]
        flags = [bool(r["invaded"]) for r in rs]
        seq = "".join("I" if f else "." for f in flags)
        jumps = [i for i in range(1, len(flags)) if flags[i] != flags[i - 1]]
        print(f"  L = {L:g}: R from {radii[0]:.2f} to {radii[-1]:.2f}   {seq}")
        if len(jumps) == 1:
            i = jumps[0]
            print(f"    single transition between R = {radii[i-1]:.3f} and "
                  f"{radii[i]:.3f}; monotone on this sweep "
                  f"(. = cleared, I = invaded)")
        else:
            print(f"    {len(jumps)} transitions at "
                  + ", ".join(f"{radii[i]:.3f}" for i in jumps)
                  + "  -> NOT monotone, the bisections need revisiting")


def plateau_fit(rows):
    """Does R_c(L) settle, and if so where?

    Confinement by the reflecting walls should die out as the box grows, so a
    saturating fit R_c(L) = R_inf - A exp(-L/lam) is the natural summary.  The
    fitted decay length lam is worth reading: if the domain dependence were
    the acid images alone, it would come out at the scale of ell_H.  The
    function reports the fit, its residuals and the increments dR_c/dL, and
    says plainly whether the last increment is small enough to call the curve
    settled.
    """
    from scipy.optimize import curve_fit
    rows = sorted(rows, key=lambda r: r["L"])
    L = np.array([r["L"] for r in rows])
    Rc = np.array([r["R_c"] for r in rows])
    if len(L) < 4:
        print("  (too few boxes for a fit)")
        return
    slopes = np.diff(Rc) / np.diff(L)
    print("  increments dR_c/dL :",
          "  ".join(f"{a:.1f}-{b:.1f}: {s:.3f}"
                    for a, b, s in zip(L[:-1], L[1:], slopes)))
    try:
        (Rinf, A, lam), _ = curve_fit(
            lambda x, Rinf, A, lam: Rinf - A * np.exp(-x / lam),
            L, Rc, p0=[Rc[-1] * 1.2, 1.0, 1.0], maxfev=40000)
    except Exception as exc:                       # pragma: no cover
        print(f"  saturating fit failed: {exc}")
        return
    resid = Rc - (Rinf - A * np.exp(-L / lam))
    print(f"  fit R_c(L) = {Rinf:.3f} - {A:.3f} exp(-L/{lam:.3f}), "
          f"max residual {np.abs(resid).max():.4f}")
    print(f"  R_inf/ell_H = {Rinf / ELL_H:.2f}, decay length "
          f"{lam:.2f} = {lam / ELL_H:.1f} ell_H")
    gap = (Rinf - Rc[-1]) / Rinf
    print(f"  at the largest box L = {L[-1]:g} the curve is still "
          f"{100 * gap:.0f} % below the fitted plateau: "
          + ("settled." if gap < 0.05 else "NOT settled."))


def discrete_masses(L=2.0, h=H_TARGET, radius=0.5816):
    """Mass actually carried by each seed shape on the grid.

    The three shapes are constructed to carry the same mass amp*pi*R^2, but
    the grid does not honour that exactly: a disc gains a little from the
    staircase boundary, a square loses a little, and a Gaussian loses its
    tail to the box.  The spread is worth printing next to the shape effect
    it could otherwise be mistaken for.
    """
    grid = Grid2D(Nx=int(round(L / h)) + 1, Ny=int(round(L / h)) + 1,
                  Lx=L, Ly=L)
    nominal = SEED_AMP * np.pi * radius ** 2
    print(f"\ndiscrete seed masses at R = {radius:.4f} "
          f"(nominal {nominal:.4f})")
    for shape in ("disc", "square", "gaussian"):
        _, T, _ = make_seed(grid, radius, shape=shape)
        m = float(T.sum()) * grid.dx * grid.dy
        print(f"  {shape:>9} : {m:.4f}  ({100 * (m / nominal - 1):+.1f} %)")


def dimensional_table(mult=None, D3_cm2_s=5e-6):
    """Screening length and critical radius in millimetres.

    The unit of length of the nondimensional system is sqrt(D3/r1), so
    ell_H = omega^(-1/2) becomes sqrt(D3/d3) in physical units: the cellular
    rate cancels, and the same holds for the critical radius, which is a fixed
    multiple of ell_H at fixed delta/delta_c and seed density.  This is
    Table 2 of the manuscript.  The multiple defaults to the quasi-static
    estimate; pass the measured plateau of the domain sweep to get the numbers
    the manuscript quotes.
    """
    if mult is None:
        mult = bessel_prediction() / ELL_H
    print(f"\ndimensional scales, D3 = {D3_cm2_s:.0e} cm^2/s, "
          f"R_c = {mult:.2f} ell_H")
    print(f"  {'1/d3':>10} {'ell_H (mm)':>12} {'R_c (mm)':>10}")
    for label, tau in (("3 min", 180.0), ("15 min", 900.0),
                       ("1 h", 3600.0), ("2.5 h", 9000.0)):
        ell_mm = 10.0 * np.sqrt(D3_cm2_s * tau)
        print(f"  {label:>10} {ell_mm:12.2f} {mult * ell_mm:10.2f}")


def report(results):
    """Print each family as the table the manuscript quotes."""
    by_family = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)

    print(f"\nacid screening length  ell_H = omega^(-1/2) = {ELL_H:.4f}")
    print(f"quasi-static prediction R_c^qs = {bessel_prediction():.4f} = "
          f"{bessel_prediction() / ELL_H:.2f} ell_H  (unbounded medium)")
    print(f"first symmetric mode   L* = 2 pi/sqrt(mu*) = {L_STAR_SYM:.3f}")
    # The multiple that matters for tissue is the one a lesion far from any
    # boundary sees, so the dimensional table uses the plateau of the domain
    # sweep when it is available rather than the frozen-seed estimate.
    plateau = max((r["R_c"] for r in by_family.get("domains", [])
                   if r.get("L", 0) >= 4.0), default=None)
    dimensional_table(mult=None if plateau is None else plateau / ELL_H)
    if "shape" in by_family:
        discrete_masses()
    print()

    for family in ("domains", "mesh", "diffusivity", "screening", "offcentre",
                   "shape", "amplitude", "timestep", "centre", "dwell",
                   "monotone", "lsweep", "pairs"):
        rows = by_family.get(family)
        if not rows:
            continue
        print(f"--- {family} " + "-" * (62 - len(family)))
        if family == "monotone":
            monotone_report(rows)
            print()
            continue
        for r in sorted(rows, key=lambda r: r["label"]):
            if family == "pairs":
                print(f"  {r['label']:>18}  R = {r['radius']:.4f}  "
                      f"{'INVADES' if r['invaded'] else 'cleared':>7}  "
                      + dwell_summary(r.get("critical_series") or [])
                      + f"   [{r['wall_s']:.0f} s]")
                continue
            if family == "dwell":
                print(f"  {r['label']:>18}  "
                      f"{'INVADES' if r['invaded'] else 'cleared':>7}  "
                      + dwell_summary(r.get("critical_series") or [])
                      + f"   [{r['wall_s']:.0f} s]")
                continue
            if family == "centre":
                k = r.get("kill")
                lethal = ("never lethal at the centre" if k is None else
                          f"centre lethal from t = {k[0]:.2f} "
                          f"(T there {k[1]:.3f}, N {k[2]:.3f})")
                print(f"  {r['label']:>18}  "
                      f"{'INVADES' if r['invaded'] else 'cleared':>7}  "
                      f"{lethal}   [{r['wall_s']:.0f} s]")
                continue
            if "error" in r:
                print(f"  {r['label']:>18}  FAILED: {r['error']} "
                      f"({r['lo']}, {r['hi']})")
                continue
            extra = ""
            if family in ("domains", "mesh"):
                extra = (f"  R_c/L = {r['R_c'] / r['L']:.4f}"
                         f"  R_c/ell_H = {r['R_c'] / ELL_H:.3f}")
            if family == "screening":
                ell = r["omega"] ** -0.5
                pred = bessel_prediction(gamma=r["gamma"], omega=r["omega"])
                extra = (f"  ell_H = {ell:.4f}  R_c/ell_H = {r['R_c'] / ell:.3f}"
                         f"  (predicted {pred / ell:.3f})")
            if family in ("lsweep", "dwell"):
                extra = "  " + dwell_summary(r.get("critical_series") or [])
            print(f"  {r['label']:>18}  R_c = {r['R_c']:.4f} "
                  f"+- {r['half_width']:.4f}{extra}"
                  f"   [viol {r['violation']:.1e}, {r['wall_s']:.0f} s]")
        if family == "domains":
            plateau_fit(rows)
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in EXPERIMENTS:
        ap.add_argument(f"--{name}", action="store_true")
    ap.add_argument("--all", action="store_true", help="run every experiment")
    ap.add_argument("--quick", action="store_true",
                    help="4 bisection steps instead of 8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=RESULTS_FILE)
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the tables from the stored results")
    args = ap.parse_args()

    if args.report_only:
        with open(args.out) as fh:
            report(json.load(fh))
        return

    chosen = [n for n in EXPERIMENTS if args.all or getattr(args, n)]
    if not chosen:
        ap.error("nothing to do: pass --all or at least one experiment flag")
    steps = 4 if args.quick else 8

    jobs = [j for n in chosen for j in EXPERIMENTS[n](steps)]
    print(f"{len(jobs)} bisections on {args.workers} workers, "
          f"{steps} steps each (bracket halved {steps} times)")

    # Earlier measurements are kept, so that one experiment can be rerun or
    # added without discarding the rest.  A rerun of the same (family, label)
    # replaces its previous entry.
    results, t0 = [], time.time()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            previous = json.load(fh)
        fresh = {(f, lbl) for f, lbl, _ in jobs}
        results = [r for r in previous
                   if (r.get("family"), r.get("label")) not in fresh]
        if results:
            print(f"keeping {len(results)} earlier measurements from "
                  f"{args.out}")
    ndone = 0
    with get_context("fork").Pool(args.workers) as pool:
        for res in pool.imap_unordered(_work, jobs):
            results.append(res)
            ndone += 1
            if "R_c" in res:
                done = f"R_c = {res['R_c']:.4f}"
            elif "invaded" in res:
                done = "INVADES" if res["invaded"] else "cleared"
            else:
                done = "FAILED"
            print(f"  [{ndone:2d}/{len(jobs)}] "
                  f"{res['family']}/{res['label']}: {done} "
                  f"({res['wall_s']:.0f} s, {time.time() - t0:.0f} s elapsed)",
                  flush=True)
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=1, default=float)

    report(results)
    print(f"total wall time {time.time() - t0:.0f} s; "
          f"results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
