"""
bifurcation_plane.py -- Phase diagram in the (alpha, delta/delta_c) plane,
plus the two transcritical bifurcation diagrams across Gamma_delta.

Produces `bifurcation_plane.pdf` (3 panels, vector, for inclusion in the
manuscript).

Panel (a): the four regions delimited by the transcritical curves
    Gamma_alpha = {alpha = 1} and Gamma_delta = {delta = delta_c},
    meeting at the codimension-two point P* = (1, 1) in rescaled
    coordinates.  The coexistence equilibrium E3 is admissible on TWO
    disjoint branches with opposite dynamical roles:
        branch (i)  alpha < 1, delta < delta_c : stable attractor
        branch (ii) alpha > 1, delta > delta_c : hyperbolic saddle,
                    whose 2D stable manifold separates the basins of
                    E1 and E2.

Panel (b): bifurcation diagram in delta/delta_c at fixed alpha = 0.3 < 1
    (E3 emerges from E2 below threshold, stable).
Panel (c): the same at fixed alpha = 1.2 > 1
    (E3 emerges from E2 above threshold, as a saddle).

Solid = stable, dashed = unstable/saddle, throughout.

Colours: Okabe-Ito, colour-vision-deficiency safe; each region also
carries a hatch texture and a direct in-region label, so identity never
rests on colour alone (print / greyscale safe).

Author: Yaya Youssouf Yaya
Date:   2026-07-11
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from params import AGGRESSIVE, INDOLENT, MODERATE, PHENOTYPES

# Embed TrueType rather than Type 3 fonts: Type 3 is rejected by most
# publisher production pipelines and makes the text non-extractable.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

# ─── Bulk parameters shared by all phenotypes (Table 2) ──────────────
BETA, GAMMA, OMEGA = MODERATE.beta, MODERATE.gamma, MODERATE.omega
DELTA_C = OMEGA / GAMMA                       # = 1.6

# Okabe-Ito, one hue per region (validated: CVD-safe, adjacent dE >= 17)
C_E1 = "#0072B2"     # blue      -- tumour-free suppression
C_E3 = "#009E73"     # green     -- coexistence
C_E2 = "#D55E00"     # vermilion -- full invasion
C_BI = "#CC79A7"     # purple    -- bistable, E3 = saddle

A_MAX, Y_MAX = 2.2, 3.4                       # axes limits, y = delta/delta_c


def n_star(alpha: float, y: np.ndarray | float) -> np.ndarray | float:
    """N* as a function of alpha and the rescaled toxicity y = delta/delta_c.

    With delta = y * delta_c = y * omega/gamma one gets delta*gamma = y*omega,
    so N* = (omega - delta gamma)/(omega - alpha delta gamma) = (1-y)/(1-alpha*y).
    """
    return (1.0 - y) / (1.0 - alpha * y)


# ════════════════════════════════════════════════════════════════════
# Panel (a) -- the parameter plane
# ════════════════════════════════════════════════════════════════════
def panel_plane(ax) -> None:
    # --- the four regions, as filled rectangles + hatch ---------------
    regions = [
        # (x0, y0, w, h, colour, hatch)
        (0.0, 0.0, 1.0, 1.0, C_E3, "/"),          # E3 coexistence
        (0.0, 1.0, 1.0, Y_MAX - 1.0, C_E2, "\\"),  # E2 invasion
        (1.0, 0.0, A_MAX - 1.0, 1.0, C_E1, "."),   # E1 suppression
        (1.0, 1.0, A_MAX - 1.0, Y_MAX - 1.0, C_BI, "x"),  # bistable
    ]
    for x0, y0, w, h, colour, hatch in regions:
        # flat tint, then a separate texture overlay so the regions stay
        # distinguishable in greyscale and under colour-vision deficiency
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=colour, alpha=0.15,
                               edgecolor="none", linewidth=0, zorder=0))
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", alpha=0.22,
                               edgecolor=colour, hatch=hatch, linewidth=0,
                               zorder=1))

    # The slice 1 < y <= 1/(1-alpha) and the wedge where the Volterra
    # criterion (HL) fails were both drawn here in earlier versions.  Neither
    # is a feature of the dynamics: the order argument of Theorem 3.x covers
    # the slice, and (HL) constrains the Lyapunov certificate, not the result.
    # Global convergence now holds on each of the three non-bistable regions
    # without qualification, so the plane is drawn without them.

    # --- the two transcritical curves and the codim-2 point ------------
    ax.axvline(1.0, color="k", lw=1.8, zorder=4)
    ax.axhline(1.0, color="k", lw=1.8, zorder=4)
    ax.plot([1.0], [1.0], "o", ms=9, mfc="w", mec="k", mew=1.8, zorder=6)
    ax.annotate(r"$P^{*}=(1,\,1)$", xy=(1.0, 1.0), xytext=(1.26, 0.70),
                fontsize=9.5, zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.9, color="k",
                                shrinkA=0, shrinkB=6))
    ax.text(2.14, 1.07, r"$\Gamma_{\delta}$", ha="right", va="bottom",
            fontsize=10, zorder=6)
    ax.text(1.05, 3.30, r"$\Gamma_{\alpha}$", ha="left", va="top",
            fontsize=10, zorder=6)

    # --- direct in-region labels (identity never rests on colour) ------
    ax.text(0.55, 0.66, "$E_3$  attracts all\ninterior data",
            ha="center", va="center", fontsize=10.5, color="#123", zorder=5)
    ax.text(0.40, 2.45, "$E_2$  attracts all\ninterior data",
            ha="center", va="center", fontsize=10.5, color="#123", zorder=5)
    ax.text(1.62, 0.55, "$E_1$  attracts all\ninterior data",
            ha="center", va="center", fontsize=10.5, color="#123", zorder=5)
    ax.text(1.62, 2.45, "$E_1\\,/\\,E_2$  bistable\n$E_3$ = saddle\n(no global result)",
            ha="center", va="center", fontsize=10.5, color="#123", zorder=5)

    # --- the three phenotypes of Table 2 -------------------------------
    for ph in PHENOTYPES:
        ax.plot([ph.alpha], [ph.delta / ph.delta_c], marker="D", ms=6.5,
                mfc="w", mec="k", mew=1.3, zorder=7)
    ax.annotate("indolent", xy=(INDOLENT.alpha, INDOLENT.delta / DELTA_C),
                xytext=(1.45, 0.16), fontsize=8.5, zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="k",
                                shrinkA=0, shrinkB=5))
    ax.annotate("moderate", xy=(MODERATE.alpha, MODERATE.delta / DELTA_C),
                xytext=(0.03, 0.86), fontsize=8.5, zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="k",
                                shrinkA=0, shrinkB=5))
    ax.annotate("aggressive", xy=(AGGRESSIVE.alpha, AGGRESSIVE.delta / DELTA_C),
                xytext=(0.05, 2.05), fontsize=8.5, zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.8, color="k",
                                shrinkA=0, shrinkB=5))

    ax.set_xlim(0, A_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel(r"competition coefficient  $\alpha$")
    ax.set_ylabel(r"rescaled acid toxicity  $\delta/\delta_c$")
    ax.set_title("(a)  parameter plane", loc="left", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ════════════════════════════════════════════════════════════════════
# Panels (b), (c) -- transcritical bifurcation across Gamma_delta
# ════════════════════════════════════════════════════════════════════
def panel_transcritical(ax, alpha: float, tag: str) -> None:
    """N-component of every equilibrium against y = delta/delta_c."""
    branch_i = alpha < 1.0

    # E3: admissible where sign(1-y) == sign(1-alpha)
    if branch_i:
        y3 = np.linspace(1e-4, 1.0, 400)          # y < 1
    else:
        y3 = np.linspace(1.0, Y_MAX, 400)         # y > 1
    n3 = n_star(alpha, y3)
    ax.plot(y3, n3, color=C_E3, lw=2.4,
            ls="-" if branch_i else (0, (5, 2)), zorder=4,
            label=r"$E_3$" + (" (stable)" if branch_i else " (saddle)"))

    # the inadmissible continuation of the same algebraic branch
    y_gh = np.linspace(1.0, Y_MAX, 400) if branch_i else np.linspace(1e-4, 1.0, 400)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_gh = n_star(alpha, y_gh)
    n_gh[np.abs(1.0 - alpha * y_gh) < 1e-3] = np.nan     # apparent singularity
    n_gh[(n_gh < -0.25) | (n_gh > 1.25)] = np.nan
    ax.plot(y_gh, n_gh, color=C_E3, lw=1.0, ls=(0, (1, 2)), alpha=0.6,
            zorder=2)

    # E2 : N = 0, stable iff y > 1
    ax.plot([0, 1], [0, 0], color=C_E2, lw=2.4, ls=(0, (5, 2)), zorder=3)
    ax.plot([1, Y_MAX], [0, 0], color=C_E2, lw=2.4, ls="-", zorder=3,
            label=r"$E_2$")

    # E1 : N = 1, stable iff alpha > 1
    ax.plot([0, Y_MAX], [1, 1], color=C_E1, lw=2.4,
            ls="-" if alpha > 1 else (0, (5, 2)), zorder=3, label=r"$E_1$")

    # the transcritical point itself
    ax.plot([1.0], [0.0], "o", ms=8, mfc="w", mec="k", mew=1.6, zorder=6)
    ax.axvline(1.0, color="k", lw=0.9, alpha=0.5, zorder=1)

    ax.set_xlim(0, Y_MAX)
    ax.set_ylim(-0.12, 1.16)
    ax.set_xlabel(r"$\delta/\delta_c$")
    ax.set_ylabel(r"normal-cell density at equilibrium,  $N$")
    ax.set_title(rf"{tag}  transcritical at $\Gamma_\delta$, "
                 rf"$\alpha={alpha}$", loc="left", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="center right", fontsize=8.5, frameon=False)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    panel_plane(axes[0])
    panel_transcritical(axes[1], alpha=0.3, tag="(b)")
    panel_transcritical(axes[2], alpha=1.2, tag="(c)")

    # one shared style key: solid = stable, dashed = unstable
    key = [Line2D([], [], color="k", lw=2.2, ls="-", label="stable"),
           Line2D([], [], color="k", lw=2.2, ls=(0, (5, 2)),
                  label="unstable / saddle"),
           Line2D([], [], color=C_E3, lw=1.0, ls=(0, (1, 2)),
                  label=r"inadmissible continuation of $E_3$")]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.035))

    fig.tight_layout()
    fig.savefig("bifurcation_plane.pdf", bbox_inches="tight")
    print("wrote bifurcation_plane.pdf")

    # --- consistency checks printed alongside the figure ---------------
    print(f"delta_c = omega/gamma = {DELTA_C}")
    for ph in PHENOTYPES:
        y = ph.delta / ph.delta_c
        adm = (ph.alpha < 1 and y < 1) or (ph.alpha > 1 and y > 1)
        print(f"  {ph.name:>10}: alpha={ph.alpha}, delta/delta_c={y:.4f}, "
              f"E3 admissible={adm}, expected regime={ph.expected_regime}")


if __name__ == "__main__":
    main()
