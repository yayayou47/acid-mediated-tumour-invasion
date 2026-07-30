"""
params.py — Single source of truth for the parameter values
used across all simulations and figures of the manuscript.

The three phenotypes match Table 2 of the manuscript exactly.
Any change here must be reflected in the manuscript (Table 2 and
the captions of Figures 3-5).

Author: Yaya Youssouf Yaya
Date:   2026-04-18
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phenotype:
    """Nondimensional parameters for one tumor phenotype.

    Attributes follow Table 2 of the manuscript.
    The strict diffusion coefficients δ₁, δ₂ are kept as small but
    NOT biologically tiny (10⁻⁵), so that fronts and interfaces
    remain numerically resolvable on a 100×100 grid in
    reasonable wall-clock time.  The biologically realistic values
    (~10⁻⁴ for cell vs H⁺) are used in the convergence study only.
    """
    name: str
    alpha: float
    beta: float
    gamma: float
    omega: float
    delta: float
    delta1: float
    delta2: float
    expected_regime: str

    @property
    def delta_c(self) -> float:
        """Critical acid-toxicity threshold ω/γ (gauge invariant)."""
        return self.omega / self.gamma


# ─── Three canonical phenotypes (Table 2 of the manuscript) ──────────
# The (γ, ω) pair is chosen so that the corrected sufficient
# condition (HL) holds for the moderate phenotype
# (with c_i = 1). The aggressive phenotype lies in the E_2 invasion
# regime, where (HL) is not relevant (global stability of E2 governs).
# The indolent phenotype illustrates the E_1-stability case (alpha > 1),
# which does not invoke the Lyapunov framework either.

INDOLENT = Phenotype(
    name="indolent",
    alpha=1.2, beta=1.5, gamma=5.0, omega=8.0,
    delta=0.5, delta1=5e-3, delta2=5e-3,
    expected_regime="E1 stable (alpha > 1)",
)

MODERATE = Phenotype(
    name="moderate",
    alpha=0.3, beta=1.5, gamma=5.0, omega=8.0,
    delta=0.5, delta1=5e-3, delta2=5e-3,
    expected_regime="E3 coexistence (delta = 0.5 < delta_c = 1.6)",
)

AGGRESSIVE = Phenotype(
    name="aggressive",
    alpha=0.3, beta=1.5, gamma=5.0, omega=8.0,
    delta=2.5, delta1=5e-3, delta2=5e-3,
    expected_regime="E2 invasion (delta = 2.5 > delta_c = 1.6)",
)

PHENOTYPES = [INDOLENT, MODERATE, AGGRESSIVE]


# ─── Common simulation grid (must match manuscript Numerics section) ─
GRID_NX, GRID_NY = 100, 100
DOMAIN_LX, DOMAIN_LY = 2.0, 2.0
DT_DEFAULT = 5e-3
T_FINAL_DEFAULT = 12.0
SEED = 42


def hl_prime(p: Phenotype) -> tuple[bool, bool, float, float]:
    """Check the Sylvester condition (HL) with c_i = 1.

    Returns (cond1_ok, cond2_ok, alpha2_beta, Delta) where
    Delta = 4 β ω - γ² - α² β² ω - α β δ γ - β δ².
    """
    a, b, g, w, d = p.alpha, p.beta, p.gamma, p.omega, p.delta
    alpha2_beta = a * a * b
    Delta = 4 * b * w - g**2 - alpha2_beta * b * w - a * b * d * g - b * d**2
    return alpha2_beta < 4, Delta > 0, alpha2_beta, Delta


if __name__ == "__main__":
    print(f"{'phenotype':>11} | {'alpha':>5} {'beta':>5} {'gamma':>6} "
          f"{'omega':>6} {'delta':>6} | {'delta_c':>8} | "
          f"{'alpha2 beta':>11} {'Delta':>8} | (HL)")
    print("-" * 100)
    for p in PHENOTYPES:
        ok1, ok2, ab, D = hl_prime(p)
        verdict = "yes" if (ok1 and ok2) else "no"
        print(f"{p.name:>11} | {p.alpha:5.2f} {p.beta:5.2f} {p.gamma:6.2f} "
              f"{p.omega:6.2f} {p.delta:6.2f} | {p.delta_c:8.4f} | "
              f"{ab:11.4f} {D:8.4f} | {verdict}")
