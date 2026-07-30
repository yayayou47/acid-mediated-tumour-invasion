"""
main.py -- Single entry point for all numerical experiments and figures
of the acid-mediated tumor invasion paper (Yaya & Mahamat).

Single external module: ``params.py`` (canonical phenotype parameters,
loaded as the unique source of truth). Everything else
(Parameters, Grid2D, Solver, InitialCondition, figure generators,
convergence study, Fisher-KPP speed measurement, three-component
overlay synthesis) is consolidated below.

Run ``python3 main.py`` to regenerate every figure used in the
manuscript. Each figure call prints its own short report.

Reproducibility: fixed seed from ``params.SEED``; default time step
``params.DT_DEFAULT``; final simulation time ``params.T_FINAL_DEFAULT``;
grid ``params.GRID_NX x params.GRID_NY`` on
``[0, params.DOMAIN_LX] x [0, params.DOMAIN_LY]``.

The model:
    dN/dt = N (1 - N - delta H)        + delta_1 Laplacian(N)
    dT/dt = beta T (1 - T - alpha N)   + delta_2 Laplacian(T)
    dH/dt = gamma T - omega H          + Laplacian(H)
with homogeneous Neumann boundary conditions and delta_3 = 1 (gauge).

Author: Yaya Youssouf Yaya
Date:   2026-04-18
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import kron, identity, diags, csc_matrix
from scipy.sparse.linalg import splu

from params import (
    INDOLENT, MODERATE, AGGRESSIVE,
    GRID_NX, GRID_NY, DOMAIN_LX, DOMAIN_LY,
    DT_DEFAULT, T_FINAL_DEFAULT, SEED,
)

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13,
                     "axes.labelsize": 13, "figure.dpi": 150})


# ══════════════════════════════════════════════════════════════════════
# 1. CORE NUMERICAL CLASSES
# ══════════════════════════════════════════════════════════════════════

class Parameters:
    """Nondimensional parameters of the acid-mediated invasion model."""

    def __init__(self, alpha=0.3, beta=1.5, gamma=5.0, delta=0.5,
                 omega=8.0, delta1=5e-3, delta2=5e-3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.delta1 = delta1
        self.delta2 = delta2

    @property
    def delta_c(self):
        return self.omega / self.gamma

    @property
    def E3(self):
        """Coexistence equilibrium, admissible on TWO disjoint branches.

        With p = omega - delta*gamma and q = omega - alpha*delta*gamma,
        the three components are simultaneously positive iff
        sign(p) = sign(q) = sign(1 - alpha), i.e. iff either
            branch (i)  alpha < 1 and delta < delta_c   -> stable node,
            branch (ii) alpha > 1 and delta > delta_c   -> saddle.
        Returns None when E3 is not admissible.
        """
        a, d, g, w = self.alpha, self.delta, self.gamma, self.omega
        denom = w - a * d * g
        if abs(denom) < 1e-14:
            return None
        N = (w - d * g) / denom
        T = w * (1 - a) / denom
        H = g * (1 - a) / denom
        if min(N, T, H) <= 0:
            return None
        return (N, T, H)

    @property
    def E3_is_saddle(self):
        """True on branch (ii): a3 = beta N* T* (omega - alpha delta gamma) < 0."""
        e3 = self.E3
        if e3 is None:
            return False
        return (self.omega - self.alpha * self.delta * self.gamma) < 0

    @property
    def E2(self):
        return (0.0, 1.0, self.gamma / self.omega)

    @property
    def regime(self):
        """Which of the four regions of the (alpha, delta/delta_c) plane."""
        if self.alpha > 1:
            return ("suppression" if self.delta < self.delta_c
                    else "bistable")          # E1/E2, E3 is the saddle
        return "coexistence" if self.delta < self.delta_c else "invasion"

    def check_HL(self):
        """Lyapunov hypothesis (HL) with c_i = 1:
        alpha^2 beta < 4  AND
        4 beta omega > gamma^2 + alpha^2 beta^2 omega
                       + alpha beta delta gamma + beta delta^2.
        """
        a, b, g, d, w = (self.alpha, self.beta, self.gamma,
                         self.delta, self.omega)
        Delta = 4*b*w - g*g - a*a*b*b*w - a*b*d*g - b*d*d
        return (a * a * b < 4) and (Delta > 0)

    def hl_details(self):
        a, b, g, d, w = (self.alpha, self.beta, self.gamma,
                         self.delta, self.omega)
        Delta = 4*b*w - g*g - a*a*b*b*w - a*b*d*g - b*d*d
        return {"alpha2_beta": a*a*b, "Delta": Delta,
                "cond_1": a*a*b < 4, "cond_2": Delta > 0}

    def __repr__(self):
        return (f"Parameters(α={self.alpha}, β={self.beta}, γ={self.gamma}, "
                f"δ={self.delta}, ω={self.omega}) → {self.regime}, "
                f"δ_c={self.delta_c:.2f}")


def from_phenotype(p):
    """Convert a Phenotype dataclass (params.py) to a Parameters."""
    return Parameters(alpha=p.alpha, beta=p.beta, gamma=p.gamma,
                      delta=p.delta, omega=p.omega,
                      delta1=p.delta1, delta2=p.delta2)


class Grid2D:
    """Uniform 2D grid on [0, Lx] x [0, Ly]."""

    def __init__(self, Nx=120, Ny=120, Lx=2.0, Ly=2.0):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Ndof = Nx * Ny
        self.extent = [0, Lx, 0, Ly]


def build_laplacian(grid):
    """Discrete 2D Laplacian with homogeneous Neumann boundary conditions."""
    Nx, Ny, dx, dy = grid.Nx, grid.Ny, grid.dx, grid.dy
    mx = -2.0 * np.ones(Nx); mx[0] = mx[-1] = -1.0
    Ax = diags([np.ones(Nx-1), mx, np.ones(Nx-1)], [-1, 0, 1],
               shape=(Nx, Nx)) / dx**2
    my = -2.0 * np.ones(Ny); my[0] = my[-1] = -1.0
    Ay = diags([np.ones(Ny-1), my, np.ones(Ny-1)], [-1, 0, 1],
               shape=(Ny, Ny)) / dy**2
    return csc_matrix(kron(identity(Ny), Ax) + kron(Ay, identity(Nx)))


class Solver:
    """IMEX time integrator: implicit diffusion (LU-factorised), explicit
    reaction. No clipping by default; pointwise violations of the
    invariant box are recorded and reported."""

    def __init__(self, grid, params, dt=5e-3, enforce_clip=False, h0_max=0.0):
        self.grid = grid
        self.params = params
        self.dt = dt
        # M0 as in Theorem 3.1: max(gamma/omega, ||H0||_inf). The caller
        # must pass h0_max = ||H0||_inf when the initial acid is nonzero;
        # every experiment in this paper starts from a localised tumour
        # seed with H0 whose sup is below gamma/omega (verified: ~0.57 <
        # 0.625), so the default h0_max = 0 gives the tight bound
        # M0 = gamma/omega = 0.625, not the looser value 1.0 used before.
        self.M0 = max(params.gamma / params.omega, h0_max)
        self.enforce_clip = enforce_clip
        self.viol = {"N_low": 0.0, "N_high": 0.0,
                     "T_low": 0.0, "T_high": 0.0,
                     "H_low": 0.0, "H_high": 0.0,
                     "n_steps": 0}
        L = build_laplacian(grid)
        I = identity(grid.Ndof, format="csc")
        self.LU_N = splu(I - dt * params.delta1 * L)
        self.LU_T = splu(I - dt * params.delta2 * L)
        self.LU_H = splu(I - dt * 1.0 * L)

    def _record_viol(self, N, T, H):
        v = self.viol
        v["N_low"] = max(v["N_low"], -float(N.min()))
        v["N_high"] = max(v["N_high"], float(N.max()) - 1.0)
        v["T_low"] = max(v["T_low"], -float(T.min()))
        v["T_high"] = max(v["T_high"], float(T.max()) - 1.0)
        v["H_low"] = max(v["H_low"], -float(H.min()))
        v["H_high"] = max(v["H_high"], float(H.max()) - self.M0)
        v["n_steps"] += 1

    def step(self, N, T, H):
        dt, p = self.dt, self.params
        Ny, Nx = self.grid.Ny, self.grid.Nx
        FN = N * (1.0 - N - p.delta * H)
        FT = p.beta * T * (1.0 - T - p.alpha * N)
        FH = p.gamma * T - p.omega * H
        N = self.LU_N.solve((N + dt * FN).ravel()).reshape(Ny, Nx)
        T = self.LU_T.solve((T + dt * FT).ravel()).reshape(Ny, Nx)
        H = self.LU_H.solve((H + dt * FH).ravel()).reshape(Ny, Nx)
        self._record_viol(N, T, H)
        if self.enforce_clip:
            np.clip(N, 0.0, 1.0, out=N)
            np.clip(T, 0.0, 1.0, out=T)
            np.clip(H, 0.0, self.M0, out=H)
        return N, T, H

    def violation_report(self):
        return dict(self.viol)

    def run(self, N0, T0, H0, T_final, snap_times=None):
        Nt = int(T_final / self.dt)
        N, T, H = N0.copy(), T0.copy(), H0.copy()
        snaps = {}
        if snap_times is None:
            snap_times = [0.0, T_final]
        for n in range(Nt + 1):
            t = n * self.dt
            for ts in snap_times:
                if abs(t - ts) < 0.5 * self.dt and ts not in snaps:
                    snaps[ts] = (N.copy(), T.copy(), H.copy())
            if n == Nt:
                break
            N, T, H = self.step(N, T, H)
        if T_final not in snaps:
            snaps[T_final] = (N.copy(), T.copy(), H.copy())
        return snaps

    def run_lyapunov(self, N0, T0, H0, T_final, sample_every=20):
        Nt = int(T_final / self.dt)
        N, T, H = N0.copy(), T0.copy(), H0.copy()
        p = self.params
        dx, dy = self.grid.dx, self.grid.dy
        eps = 1e-12
        e3 = p.E3
        tt, vL = [], []
        for n in range(Nt + 1):
            if n % sample_every == 0:
                if e3 is not None:
                    Ns, Ts, Hs = e3
                    Nn = np.maximum(N, eps); Tn = np.maximum(T, eps)
                    LN = np.sum(Nn - Ns - Ns * np.log(Nn / Ns)) * dx * dy
                    LT = np.sum(Tn - Ts - Ts * np.log(Tn / Ts)) * dx * dy
                    LH = 0.5 * np.sum((H - Hs)**2) * dx * dy
                else:
                    Hs = p.gamma / p.omega
                    Tn = np.maximum(T, eps)
                    LN = 0.5 * np.sum(N**2) * dx * dy
                    LT = np.sum(Tn - 1 - np.log(Tn)) * dx * dy
                    LH = 0.5 * np.sum((H - Hs)**2) * dx * dy
                tt.append(n * self.dt); vL.append(LN + LT + LH)
            if n == Nt:
                break
            N, T, H = self.step(N, T, H)
        return np.array(tt), np.array(vL)


class InitialCondition:
    """Random Gaussian foci on a 2D grid (deterministic given seed)."""

    def __init__(self, grid, seed=42):
        self.grid = grid
        self.rng = np.random.RandomState(seed)

    def _gaussian_field(self, n, margin=0.3,
                        amp_range=(0.6, 1.0), sig_range=(0.06, 0.13)):
        X, Y = self.grid.X, self.grid.Y
        Lx, Ly = self.grid.Lx, self.grid.Ly
        cx = self.rng.uniform(margin, Lx - margin, n)
        cy = self.rng.uniform(margin, Ly - margin, n)
        sig = self.rng.uniform(*sig_range, n)
        amp = self.rng.uniform(*amp_range, n)
        field = np.zeros_like(X)
        for i in range(n):
            field += amp[i] * np.exp(
                -((X - cx[i])**2 + (Y - cy[i])**2) / (2 * sig[i]**2)
            )
        return np.clip(field, 0, 1)

    def random_foci(self, n_tumors=5, n_acid=6):
        T0 = self._gaussian_field(n_tumors)
        H0 = self._gaussian_field(n_acid,
                                  amp_range=(0.2, 0.6),
                                  sig_range=(0.08, 0.16))
        N0 = np.clip(1.0 - T0 - 0.3 * H0, 0, 1)
        return N0, T0, H0

    def tiny_seed(self, n_tumors=5, n_acid=6, scale=0.05):
        N0_f, T0_f, H0_f = self.random_foci(n_tumors, n_acid)
        T0 = scale * T0_f
        H0 = 0.02 * H0_f
        N0 = np.clip(1.0 - T0, 0, 1)
        return N0, T0, H0


# ══════════════════════════════════════════════════════════════════════
# 2. SHARED GRID AND INITIAL DATA
# ══════════════════════════════════════════════════════════════════════

GRID = Grid2D(Nx=GRID_NX, Ny=GRID_NY, Lx=DOMAIN_LX, Ly=DOMAIN_LY)
DT = DT_DEFAULT
T_FINAL = T_FINAL_DEFAULT

IC = InitialCondition(GRID, seed=SEED)
N0_FULL, T0_FULL, H0_FULL = IC.random_foci(n_tumors=5, n_acid=6)
N0_TINY, T0_TINY, H0_TINY = IC.tiny_seed(n_tumors=5, n_acid=6)


# ══════════════════════════════════════════════════════════════════════
# 3. FIGURE 1 -- THREE-COMPONENT OVERLAY (Option 3 of the discussion):
#    one panel per regime, T as background colormap, N=0.5 isoline,
#    H = gamma/(2 omega) isoline, all on the same axes.
# ══════════════════════════════════════════════════════════════════════

def fig_regimes_overlay():
    """Single-figure synthesis of the three regimes E1, E3, E2.

    Each panel shows the tumor density T as a perceptually uniform
    background (with per-panel adaptive scale, so that small foci
    in E1 and the front in E2 remain visible), an orange T=0.5
    isoline (the explicit tumor marker), the normal-cell
    threshold {N = 0.5} (white solid), and the half-equilibrium
    acid contour {H = gamma/(2 omega)} (cyan dashed).
    The snapshot is taken at t = 6 (transient regime), which
    avoids the visual saturation of t = 12.
    """
    print("=" * 60)
    print("Figure: three-regime overlay (T colormap + N, T, H isolines)")
    print("=" * 60)

    snap_time = 6.0
    scenarios = [
        (INDOLENT,   N0_TINY, T0_TINY, H0_TINY, r"$E_1$ (suppression)"),
        (MODERATE,   N0_FULL, T0_FULL, H0_FULL, r"$E_3$ (coexistence)"),
        (AGGRESSIVE, N0_FULL, T0_FULL, H0_FULL, r"$E_2$ (invasion)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             constrained_layout=True)
    for ax, (pheno, N0, T0, H0, label) in zip(axes, scenarios):
        params = from_phenotype(pheno)
        solver = Solver(GRID, params, DT, enforce_clip=False)
        snaps = solver.run(N0, T0, H0, snap_time, [snap_time])
        N, T, H = snaps[snap_time]
        H_iso = pheno.gamma / (2.0 * pheno.omega)

        # Per-panel adaptive scale to keep the tumor visible:
        # E1 has T_max ~ 0.05, E3 ~ 0.8, E2 ~ 1.
        vmax_T = max(0.05, float(T.max()))
        im = ax.imshow(T, extent=GRID.extent, origin="lower",
                       cmap="magma", vmin=0.0, vmax=vmax_T,
                       aspect="equal")
        cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
        cbar.set_label(rf"$T$ (max $\approx {vmax_T:.2f}$)",
                       fontsize=9)

        # Orange solid: explicit tumor marker (front).
        # Only drawn if the level lies inside the actual T-range.
        T_level = 0.5 * vmax_T  # half-of-current-max, always present
        ax.contour(GRID.X, GRID.Y, T, levels=[T_level],
                   colors="orange", linewidths=2.2)
        # White solid: critical normal-cell density
        ax.contour(GRID.X, GRID.Y, N, levels=[0.5],
                   colors="white", linewidths=2.0)
        # Cyan dashed: half the equilibrium acid concentration
        ax.contour(GRID.X, GRID.Y, H, levels=[H_iso],
                   colors="cyan", linewidths=1.6, linestyles="--")

        ax.set_title(
            label + "\n"
            rf"$\alpha={pheno.alpha:.1f},\ \delta={pheno.delta:.1f},"
            rf"\ \delta_c={pheno.delta_c:.1f}$",
            fontsize=11)
        ax.set_xlabel(r"$x$ (nondim.)")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
    axes[0].set_ylabel(r"$y$ (nondim.)")

    fig.suptitle(
        rf"Three regimes simultaneously simulated on $\Omega = [0,2]^2$"
        rf" at $t = {snap_time:.0f}$.  "
        r"\textbf{Orange}: $\{T = T_{\max}/2\}$ (tumor front);  "
        r"\textbf{White}: $\{N = 0.5\}$;  "
        r"\textbf{Cyan dashed}: $\{H = \gamma/(2\omega)\}$.",
        fontsize=11, fontweight="bold", y=1.06)
    fig.savefig("regimes_overlay.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> regimes_overlay.pdf\n")


# ══════════════════════════════════════════════════════════════════════
# 4. FIGURE 2 -- BIFURCATION DIAGRAM
# ══════════════════════════════════════════════════════════════════════

def fig_bifurcation():
    print("=" * 60)
    print("Figure: bifurcation diagram (mean N vs delta)")
    print("=" * 60)

    p = MODERATE  # canonical alpha, beta, gamma, omega
    delta_c = p.omega / p.gamma
    delta_values = np.concatenate([
        np.linspace(0.1, delta_c - 0.3, 8),
        np.linspace(delta_c - 0.25, delta_c + 0.25, 8),
        np.linspace(delta_c + 0.3, 4.0, 6),
    ])

    grid_b = Grid2D(Nx=60, Ny=60, Lx=DOMAIN_LX, Ly=DOMAIN_LY)
    ic_b = InitialCondition(grid_b, seed=SEED)
    N0b, T0b, H0b = ic_b.random_foci(n_tumors=5, n_acid=6)

    N_final = []
    for dv in delta_values:
        params = Parameters(alpha=p.alpha, beta=p.beta, gamma=p.gamma,
                            delta=dv, omega=p.omega,
                            delta1=p.delta1, delta2=p.delta2)
        solver = Solver(grid_b, params, dt=8e-3)
        snaps = solver.run(N0b, T0b, H0b, 18.0, [18.0])
        N_final.append(float(np.mean(snaps[18.0][0])))
        print(f"  δ={dv:.2f}  ⟨N⟩={N_final[-1]:.3f}")
    delta_values = np.array(delta_values); N_final = np.array(N_final)

    d_th = np.linspace(0.01, 4.0, 300)
    a, g, w = p.alpha, p.gamma, p.omega
    N_th = np.where(d_th < w / g,
                    (w - d_th * g) / (w - a * d_th * g), 0.0)
    N_th = np.clip(N_th, 0.0, None)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    mask = d_th <= w / g
    ax.plot(d_th[mask], N_th[mask], "b-", lw=2.5,
            label=rf"$N^*(\delta)$ theory (stable, $\alpha={a}$)")
    ax.plot(d_th[~mask], N_th[~mask], "b--", lw=1.5, alpha=0.4)
    ax.plot(delta_values, N_final, "ko", ms=8, zorder=5,
            label=r"Numerical $\langle N \rangle_{t=18}$")
    ax.axvline(delta_c, ls=":", color="grey", alpha=0.7)
    ax.text(delta_c + 0.05, 0.85,
            rf"$\delta_c = \omega/\gamma = {delta_c:.1f}$" "\n(bifurcation)",
            fontsize=10, color="grey")
    ax.set_xlabel(r"Acid toxicity $\delta$", fontsize=13)
    ax.set_ylabel(r"Equilibrium $\langle N \rangle$", fontsize=13)
    ax.set_title(
        rf"Transcritical bifurcation ($\alpha={a}$, $\omega={w}$, $\gamma={g}$)",
        fontsize=14, fontweight="bold")
    ax.set_xlim(0, 4.1); ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bifurcation_diagram.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> bifurcation_diagram.pdf\n")


# ══════════════════════════════════════════════════════════════════════
# 5. FIGURE 3 -- LYAPUNOV DECAY FOR MULTIPLE delta
# ══════════════════════════════════════════════════════════════════════

def fig_lyapunov_multi():
    print("=" * 60)
    print("Figure: Lyapunov functional decay")
    print("=" * 60)

    p = MODERATE
    delta_c = p.omega / p.gamma
    delta_list = [0.3, 0.8, 1.2, delta_c, 2.0, 3.5]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(delta_list)))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for dv, col in zip(delta_list, colors):
        print(f"  δ={dv:.2f}")
        params = Parameters(alpha=p.alpha, beta=p.beta, gamma=p.gamma,
                            delta=dv, omega=p.omega,
                            delta1=p.delta1, delta2=p.delta2)
        solver = Solver(GRID, params, DT)
        tt, Lv = solver.run_lyapunov(
            N0_FULL.copy(), T0_FULL.copy(), H0_FULL.copy(), T_FINAL)
        ls = "-" if dv < delta_c else ("--" if dv > delta_c else "-.")
        ax.semilogy(tt, Lv, color=col, lw=2, ls=ls,
                    label=rf"$\delta={dv:.1f}$")
    ax.set_xlabel(r"Time $t$", fontsize=13)
    ax.set_ylabel(r"$\mathcal{L}(t)$", fontsize=13)
    ax.set_title(
        rf"Lyapunov decay ($\delta_c = \omega/\gamma = {delta_c:.1f}$)",
        fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lyapunov_multi_delta.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> lyapunov_multi_delta.pdf\n")


# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE 4 -- CONVERGENCE STUDY
# ══════════════════════════════════════════════════════════════════════

def _coarsen(field, factor):
    return field[::factor, ::factor]


def _run_to_T(Nx, dt, T_end, p):
    g = Grid2D(Nx=Nx, Ny=Nx, Lx=DOMAIN_LX, Ly=DOMAIN_LY)
    ic = InitialCondition(g, seed=SEED)
    N0, T0, H0 = ic.random_foci(n_tumors=5, n_acid=6)
    s = Solver(g, p, dt=dt, enforce_clip=False)
    snaps = s.run(N0, T0, H0, T_end, snap_times=[T_end])
    return (*snaps[T_end], s.violation_report())


def fig_convergence():
    print("=" * 60)
    print("Figure: numerical convergence (mesh + time)")
    print("=" * 60)

    p = from_phenotype(MODERATE)
    # Grids of the form 2^k + 1 so that the coarse nodes are EXACTLY a
    # subset of the fine ones: with (50, 100, 200) the intervals are
    # 49, 99, 199, which are not nested, and `_coarsen` would then compare
    # fields at slightly different locations.
    Nx_list = (51, 101, 201, 401, 801)
    dt_list = (2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3)
    T_end = 4.0

    # Mesh refinement at fixed dt = 2.5e-3
    print("Mesh refinement at dt = 2.5e-3")
    fields_h = {}
    for Nx in Nx_list:
        Nf, Tf, Hf, viol = _run_to_T(Nx, 2.5e-3, T_end, p)
        fields_h[Nx] = (Nf, Tf, Hf)
        worst = max(viol[k] for k in viol if k != "n_steps")
        print(f"  Nx={Nx:3d}  worst pointwise violation = {worst:.2e}")
    mesh_err = []
    for i in range(len(Nx_list) - 1):
        coarse = fields_h[Nx_list[i]]
        factor = (Nx_list[i + 1] - 1) // (Nx_list[i] - 1)
        assert (Nx_list[i + 1] - 1) == factor * (Nx_list[i] - 1), \
            "grids must be nested for a meaningful pairwise comparison"
        fine_sub = tuple(_coarsen(f, factor) for f in fields_h[Nx_list[i + 1]])
        mesh_err.append(max(np.max(np.abs(c - fs))
                            for c, fs in zip(coarse, fine_sub)))

    # Time refinement at fixed Nx = 100
    print("Time refinement at Nx = 101")
    fields_t = {}
    for dt in dt_list:
        Nt = int(round(T_end / dt))
        T_eff = Nt * dt
        Nf, Tf, Hf, viol = _run_to_T(101, dt, T_eff, p)
        fields_t[dt] = (Nf, Tf, Hf)
        worst = max(viol[k] for k in viol if k != "n_steps")
        print(f"  dt={dt:g}  worst pointwise violation = {worst:.2e}")
    time_err = []
    for i in range(len(dt_list) - 1):
        coarse = fields_t[dt_list[i]]; fine = fields_t[dt_list[i + 1]]
        time_err.append(max(np.max(np.abs(c - f))
                            for c, f in zip(coarse, fine)))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    h = [DOMAIN_LX / (n - 1) for n in Nx_list[:-1]]
    axes[0].loglog(h, mesh_err, "o-", lw=2, ms=9, color="#0072B2",
                   label=r"measured $\|u_h - u_{h/2}\|_{\infty}$")
    axes[0].loglog(h, [mesh_err[-1] * (hh / h[-1])**1 for hh in h],
                   "k--", lw=1.2, label=r"slope 1")
    axes[0].loglog(h, [mesh_err[-1] * (hh / h[-1])**2 for hh in h],
                   "k:", lw=1.2, label=r"slope 2")
    for hh, e, n in zip(h, mesh_err, Nx_list[:-1]):
        axes[0].annotate(rf"$N_x={n}$", (hh, e), textcoords="offset points",
                         xytext=(6, -11), fontsize=8, color="#444")
    axes[0].set_xlabel(r"mesh step $h$")
    axes[0].set_ylabel(r"$\|u_h - u_{h/2}\|_{L^\infty}$ at $t = 4$")
    axes[0].set_title("(a) mesh refinement, $\\Delta t = 2.5\\cdot 10^{-3}$",
                      loc="left")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=9, frameon=False)

    dt_x = list(dt_list[:-1])
    axes[1].loglog(dt_x, time_err, "s-", lw=2, ms=9, color="#D55E00",
                   label=r"measured $\|u_{\Delta t} - u_{\Delta t/2}\|_{\infty}$")
    axes[1].loglog(dt_x, [time_err[-1] * (d / dt_x[-1]) for d in dt_x],
                   "k--", lw=1.2, label=r"slope 1")
    for dd, e in zip(dt_x, time_err):
        axes[1].annotate(rf"$\Delta t={dd:g}$", (dd, e),
                         textcoords="offset points", xytext=(6, -11),
                         fontsize=8, color="#444")
    axes[1].set_xlabel(r"time step $\Delta t$")
    axes[1].set_ylabel(r"$\|u_{\Delta t} - u_{\Delta t/2}\|_{L^\infty}$")
    axes[1].set_title(r"(b) time refinement, $N_x = 101$", loc="left")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig("convergence_study.pdf", dpi=200, bbox_inches="tight")
    plt.close()

    for i in range(len(mesh_err) - 1):
        print(f"  observed mesh order  ({Nx_list[i]}->{Nx_list[i+2]}) "
              f"≈ {np.log2(mesh_err[i] / mesh_err[i+1]):.3f}")
    for i in range(len(time_err) - 1):
        print(f"  observed time order  ({dt_list[i]:g}->{dt_list[i+2]:g}) "
              f"≈ {np.log2(time_err[i] / time_err[i+1]):.3f}")
    print("  -> convergence_study.pdf\n")


# ══════════════════════════════════════════════════════════════════════
# 7. FIGURE 5 -- FISHER-KPP FRONT SPEED
# ══════════════════════════════════════════════════════════════════════

def fig_2d_bifurcation():
    """Numerical bifurcation map in the (alpha, delta) plane.

    Background heat map: long-time spatial average <N>_T_end on a
    coarse parameter sweep. Overlaid: the two analytical
    transcritical curves Gamma_delta = {delta = delta_c} and
    Gamma_alpha = {alpha = 1}, the codimension-2 point
    P* = (1, delta_c), and labels of the four regions.
    """
    print("=" * 60)
    print("Figure: 2D bifurcation diagram in (alpha, delta) plane")
    print("=" * 60)

    p = MODERATE  # canonical (beta, gamma, omega) shared by the sweep
    delta_c = p.omega / p.gamma
    alphas = np.linspace(0.05, 1.6, 14)
    deltas = np.linspace(0.1, 3.5, 14)
    N_inf = np.full((len(alphas), len(deltas)), np.nan)

    grid_b = Grid2D(Nx=40, Ny=40, Lx=DOMAIN_LX, Ly=DOMAIN_LY)
    ic_b = InitialCondition(grid_b, seed=SEED)
    N0b, T0b, H0b = ic_b.random_foci(n_tumors=5, n_acid=6)
    T_end = 12.0; dt = 8e-3

    for i, a in enumerate(alphas):
        for j, dv in enumerate(deltas):
            params = Parameters(alpha=a, beta=p.beta, gamma=p.gamma,
                                delta=dv, omega=p.omega,
                                delta1=p.delta1, delta2=p.delta2)
            solver = Solver(grid_b, params, dt=dt, enforce_clip=False)
            snaps = solver.run(N0b, T0b, H0b, T_end, [T_end])
            N_inf[i, j] = float(np.mean(snaps[T_end][0]))
        print(f"  alpha = {a:.2f} done")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    A, D = np.meshgrid(alphas, deltas, indexing="ij")
    pcm = ax.pcolormesh(A, D, N_inf, cmap="viridis",
                        vmin=0.0, vmax=1.0, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"Long-time average $\langle N \rangle_{t = 12}$")

    # Analytical transcritical curves
    ax.axhline(delta_c, color="red", lw=2.0, label=r"$\Gamma_{\delta}: \delta = \delta_c$")
    ax.axvline(1.0, color="white", lw=2.0, label=r"$\Gamma_{\alpha}: \alpha = 1$")
    # Codim-2 point P*
    ax.plot([1.0], [delta_c], "o", ms=14, mfc="yellow",
            mec="black", mew=2, label=r"$P^* = (1,\delta_c)$ (codim-2)")
    # Strengthened invasion threshold delta = delta_c / (1 - alpha)
    a_grid = np.linspace(0.05, 0.99, 200)
    dline = delta_c / (1.0 - a_grid)
    ax.plot(a_grid, np.minimum(dline, 3.5), "--",
            color="orange", lw=1.5,
            label=r"$\delta = \delta_c / (1-\alpha)$ (Thm.~3.10)")

    # Region labels
    txt_kw = dict(fontsize=10, fontweight="bold",
                  ha="center", color="white",
                  bbox=dict(boxstyle="round,pad=0.25",
                            facecolor="black", alpha=0.55,
                            edgecolor="none"))
    ax.text(0.45, 0.7, r"$E_3$", **txt_kw)
    ax.text(0.45, 2.5, r"$E_2$", **txt_kw)
    ax.text(1.35, 0.7, r"$E_1$", **txt_kw)
    ax.text(1.35, 2.5, r"$E_1 / E_2$" "\n" "bistable", **txt_kw)

    ax.set_xlabel(r"competition coefficient $\alpha$", fontsize=12)
    ax.set_ylabel(r"acid toxicity $\delta$", fontsize=12)
    ax.set_title(
        rf"2D bifurcation map ($\beta = {p.beta}, \gamma = {p.gamma}, "
        rf"\omega = {p.omega}, \delta_c = {delta_c:.1f}$)",
        fontsize=12, fontweight="bold")
    ax.set_xlim(alphas[0], alphas[-1]); ax.set_ylim(deltas[0], deltas[-1])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("bifurcation_2d.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> bifurcation_2d.pdf\n")


# ══════════════════════════════════════════════════════════════════════
# 9. PROPOSED FIGURE B -- TIME EVOLUTION OF MACROSCOPIC OBSERVABLES
# ══════════════════════════════════════════════════════════════════════

def _run_means(solver, N0, T0, H0, T_final, sample_every=20):
    Nt = int(T_final / solver.dt)
    N, T, H = N0.copy(), T0.copy(), H0.copy()
    tt, mN, mT, mH = [], [], [], []
    for n in range(Nt + 1):
        if n % sample_every == 0:
            tt.append(n * solver.dt)
            mN.append(float(np.mean(N)))
            mT.append(float(np.mean(T)))
            mH.append(float(np.mean(H)))
        if n == Nt:
            break
        N, T, H = solver.step(N, T, H)
    return (np.array(tt), np.array(mN),
            np.array(mT), np.array(mH))


def fig_macroscopic_evolution():
    """Time evolution of <N>(t), <T>(t), <H>(t) for the three
    phenotypes, with the predicted equilibrium values shown as
    horizontal dashed lines.
    """
    print("=" * 60)
    print("Figure: macroscopic observables <N>, <T>, <H> vs time")
    print("=" * 60)

    scenarios = [
        (INDOLENT,   N0_TINY, T0_TINY, H0_TINY, r"$E_1$ regime"),
        (MODERATE,   N0_FULL, T0_FULL, H0_FULL, r"$E_3$ regime"),
        (AGGRESSIVE, N0_FULL, T0_FULL, H0_FULL, r"$E_2$ regime"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             sharey=True, constrained_layout=True)
    for ax, (pheno, N0, T0, H0, label) in zip(axes, scenarios):
        params = from_phenotype(pheno)
        solver = Solver(GRID, params, DT, enforce_clip=False)
        tt, mN, mT, mH = _run_means(solver, N0, T0, H0, T_FINAL)

        ax.plot(tt, mN, color="#1f77b4", lw=2.2, label=r"$\langle N \rangle$")
        ax.plot(tt, mT, color="#d62728", lw=2.2, label=r"$\langle T \rangle$")
        ax.plot(tt, mH / max(pheno.gamma / pheno.omega, 1e-9), color="#2ca02c",
                lw=2.2, ls="-",
                label=r"$\langle H \rangle / (\gamma/\omega)$")

        # Predicted equilibrium values
        if pheno.alpha >= 1:
            eqN, eqT, eqH = 1.0, 0.0, 0.0
        elif pheno.delta < pheno.delta_c:
            denom = pheno.omega - pheno.alpha * pheno.delta * pheno.gamma
            eqN = (pheno.omega - pheno.delta * pheno.gamma) / denom
            eqT = pheno.omega * (1 - pheno.alpha) / denom
            eqH = pheno.gamma * (1 - pheno.alpha) / denom
        else:
            eqN, eqT, eqH = 0.0, 1.0, pheno.gamma / pheno.omega

        for v, c in zip([eqN, eqT, eqH / (pheno.gamma / pheno.omega)
                         if pheno.gamma > 0 else 0.0],
                        ["#1f77b4", "#d62728", "#2ca02c"]):
            ax.axhline(v, color=c, ls=":", lw=1.2, alpha=0.7)

        ax.set_xlabel(r"time $t$ (nondim.)")
        ax.set_title(label + "\n" rf"($\alpha = {pheno.alpha:.1f},"
                     rf"\ \delta = {pheno.delta:.1f}$)",
                     fontsize=11)
        ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.15)
    axes[0].set_ylabel(r"spatial average")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle(
        r"Macroscopic observables $\langle N \rangle, \langle T \rangle,"
        r"\langle H \rangle / (\gamma/\omega)$ vs time."
        r"  Dotted lines: predicted equilibrium values.",
        fontsize=11, fontweight="bold", y=1.06)
    fig.savefig("macroscopic_evolution.pdf", dpi=200,
                bbox_inches="tight")
    plt.close()
    print("  -> macroscopic_evolution.pdf\n")


# ══════════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Acid-mediated tumor invasion -- regenerating all manuscript figures")
    print("=" * 70 + "\n")
    fig_regimes_overlay()
    fig_bifurcation()
    fig_lyapunov_multi()
    fig_convergence()
    # The Fisher-KPP front speed has its own script, `fisher_kpp.py`
    # (much finer mesh, much longer run); run it separately.
    fig_2d_bifurcation()
    fig_macroscopic_evolution()
    print("=" * 70)
    print("All figures generated.")
    print("=" * 70)
