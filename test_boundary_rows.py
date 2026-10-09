"""Non-regression test for the Neumann boundary rows of `main.py`.

Exit code 0 when every check passes, 1 otherwise.  Run with
    python test_boundary_rows.py

The production operator must carry the ghost-node row 2(u_1 - u_0)/h^2,
annihilate constants, and satisfy the discrete divergence theorem
sum_i w_i (A u)_i = 0 for the trapezoidal weights w = (1/2, 1, ..., 1, 1/2).
The control variant behind BOUNDARY_HALVED must carry (u_1 - u_0)/h^2,
which is what the manuscript says it is; in R1 it carried (u_1 - 2 u_0)/h^2,
and the third check is the one that catches that slip.

Author: Yaya Youssouf Yaya
Date:   2026-10-09
"""
from __future__ import annotations

import sys

import numpy as np

import main

N_NODES, H = 9, 0.25
RNG = np.random.default_rng(42)


def _rows(halved: bool) -> np.ndarray:
    main.BOUNDARY_HALVED = halved
    try:
        return main._tridiagonal_neumann(N_NODES, H).toarray() * H**2
    finally:
        main.BOUNDARY_HALVED = False


def run() -> list[tuple[str, bool]]:
    A = _rows(False)
    B = _rows(True)
    u = RNG.standard_normal(N_NODES)
    w = np.ones(N_NODES)
    w[0] = w[-1] = 0.5
    first = np.zeros(N_NODES)
    last = np.zeros(N_NODES)
    checks = []
    first[:2] = (-2.0, 2.0)
    last[-2:] = (2.0, -2.0)
    checks.append(("production rows are 2(u_1 - u_0)/h^2",
                   np.array_equal(A[0], first) and np.array_equal(A[-1], last)))
    checks.append(("production operator annihilates constants",
                   np.abs(A @ np.ones(N_NODES)).max() < 1e-14))
    first[:2] = (-1.0, 1.0)
    last[-2:] = (1.0, -1.0)
    checks.append(("control rows are (u_1 - u_0)/h^2",
                   np.array_equal(B[0], first) and np.array_equal(B[-1], last)))
    checks.append(("control operator annihilates constants",
                   np.abs(B @ np.ones(N_NODES)).max() < 1e-14))
    checks.append(("discrete divergence theorem, production",
                   abs(w @ (A @ u)) < 1e-12))
    checks.append(("interior rows untouched by the flag",
                   np.array_equal(A[1:-1], B[1:-1])))
    return checks


if __name__ == "__main__":
    ok = True
    for name, passed in run():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    sys.exit(0 if ok else 1)
