"""
Vérification symbolique des calculs du manuscrit
================================================
- Equilibres E0, E1, E2, E3
- Jacobiennes
- Polynôme caractéristique de J3 et coefficients RH
- Test de Routh-Hurwitz a1 a2 - a3 (signe et facteur)
- Test de la condition (HL) via Sylvester sur la matrice -Q
"""
import sympy as sp

# Symboles
N, T, H = sp.symbols('N T H', real=True, positive=True)
alpha, beta, gamma, delta, omega = sp.symbols('alpha beta gamma delta omega',
                                              real=True, positive=True)
delta1, delta2 = sp.symbols('delta1 delta2', real=True, positive=True)

# Champ de réaction F (sans diffusion)
F1 = N * (1 - N - delta * H)
F2 = beta * T * (1 - T - alpha * N)
F3 = gamma * T - omega * H
F = sp.Matrix([F1, F2, F3])

print("="*72)
print("1. EQUILIBRES")
print("="*72)
solutions = sp.solve([F1, F2, F3], [N, T, H], dict=True)
for k, sol in enumerate(solutions):
    print(f"  Equilibre {k}: N={sp.simplify(sol[N])}, "
          f"T={sp.simplify(sol[T])}, H={sp.simplify(sol[H])}")

# Composantes de E3 du manuscrit
N_star = (omega - delta*gamma) / (omega - alpha*delta*gamma)
T_star = omega*(1 - alpha) / (omega - alpha*delta*gamma)
H_star = gamma*(1 - alpha) / (omega - alpha*delta*gamma)

# Vérification que (N*, T*, H*) annule F
print("\nVérification que E3 annule F :")
check = sp.simplify(F.subs({N: N_star, T: T_star, H: H_star}))
print("F(E3) =", check.T)

print("\n"+"="*72)
print("2. JACOBIENNE GENERALE")
print("="*72)
J = F.jacobian([N, T, H])
print(J)

print("\n"+"="*72)
print("3. JACOBIENNE EN E3 (avec relations equilibre)")
print("="*72)
J3 = J.subs({N: N_star, T: T_star, H: H_star})
# Application des relations 1 - N* - delta H* = 0  et  1 - T* - alpha N* = 0
# -> 1 - 2 N* - delta H* = -N*    et   beta(1 - 2T* - alpha N*) = -beta T*
# On les impose par substitution simplifiante
# Méthode : reformuler la Jacobienne en termes simplifiés
J3_simplified = sp.Matrix([
    [-N_star,           sp.Integer(0),     -delta*N_star],
    [-alpha*beta*T_star, -beta*T_star,      sp.Integer(0)],
    [sp.Integer(0),      gamma,            -omega]
])

# Vérification par substitution explicite
diff_full = sp.simplify(J3 - J3_simplified)
print("Différence J3_sym - J3_simplifiée :")
print(diff_full)

print("\n"+"="*72)
print("4. POLYNOME CARACTERISTIQUE DE J3")
print("="*72)
lam = sp.symbols('lambda')
poly = sp.expand((J3_simplified - lam*sp.eye(3)).det())
poly_collected = sp.collect(sp.expand(-poly), lam)  # change signe pour forme +λ^3 + ...
print("p(λ) = λ^3 + a1 λ^2 + a2 λ + a3 avec :")
# det(J - λI) = -λ^3 + tr λ^2 - ... ; on prend p(λ) = (-1)^3 det(J-λI) = λ^3 - tr λ^2 + ...
# Plus simple : p(λ) = det(λI - J)
poly2 = sp.expand((lam*sp.eye(3) - J3_simplified).det())
poly2 = sp.collect(poly2, lam)
print("p(λ) = ", poly2)

a1 = poly2.coeff(lam, 2)
a2 = poly2.coeff(lam, 1)
a3 = poly2.coeff(lam, 0)
print("\na1 =", sp.simplify(a1))
print("a2 =", sp.simplify(a2))
print("a3 =", sp.simplify(a3))

# Comparaison avec l'expression du manuscrit
a1_man = N_star + beta*T_star + omega
a2_man = N_star*beta*T_star + N_star*omega + beta*T_star*omega
a3_man = beta*N_star*T_star*(omega - alpha*delta*gamma)

print("\nComparaison avec le manuscrit :")
print(" a1 - a1_man =", sp.simplify(a1 - a1_man))
print(" a2 - a2_man =", sp.simplify(a2 - a2_man))
print(" a3 - a3_man =", sp.simplify(a3 - a3_man))

print("\n"+"="*72)
print("5. ROUTH-HURWITZ : a1*a2 - a3")
print("="*72)
RH = sp.expand(a1_man*a2_man - a3_man)
print("a1*a2 - a3 (developpé) =")
print(sp.collect(RH, [N_star, T_star]))

# Mise en évidence : tous les termes sont positifs ?
# On suppose alpha < 1, delta < omega/gamma : alors omega - alpha*delta*gamma > 0
# et N*, T* > 0
print("\nFactorisation simplifiée :")
RH_simp = sp.simplify(RH)
print(RH_simp)

print("\n"+"="*72)
print("6. CONDITION (HL) : Sylvester sur -Q")
print("="*72)
# Matrice Q (manuscrit)
c1, c2, c3 = sp.symbols('c1 c2 c3', positive=True)
Q = sp.Matrix([
    [-c1,             -c2*alpha*beta/2, -c1*delta/2],
    [-c2*alpha*beta/2, -c2*beta,         c3*gamma/2],
    [-c1*delta/2,      c3*gamma/2,      -c3*omega]
])
mQ = -Q

# Mineurs principaux de -Q
m1 = mQ[0, 0]
m2 = (mQ[:2, :2]).det()
m3 = mQ.det()

print(f"Mineur 1 = c1 = {m1}")
print(f"Mineur 2 = {sp.simplify(m2)}")
print(f"Mineur 3 = {sp.factor(sp.simplify(m3))}")

# Cas c1=c2=c3=1
print("\nAvec c1 = c2 = c3 = 1 :")
m1_1 = m1.subs({c1:1, c2:1, c3:1})
m2_1 = sp.simplify(m2.subs({c1:1, c2:1, c3:1}))
m3_1 = sp.simplify(m3.subs({c1:1, c2:1, c3:1}))
print(f"  m1 = {m1_1}")
print(f"  m2 = {m2_1}")
print(f"  m3 = {m3_1}")
print(f"  m3 factorisé = {sp.factor(m3_1)}")

# L'hypothese (HL) du manuscrit est exactement m3 > 0 avec c1 = c2 = c3 = 1,
# c'est-a-dire Delta > 0 pour
# Delta = 4 beta omega - gamma^2 - alpha^2 beta^2 omega - alpha beta delta gamma
#         - beta delta^2,
# et elle est verifiee ci-dessus par le mineur m3 = Delta/4.  Une version
# anterieure de ce script imprimait ici une autre expression, heritee d'un
# etat depasse du manuscrit ; elle a ete retiree pour qu'aucune formule
# etiquetee « du manuscrit » ne circule sans y correspondre.
Delta_man = 4*beta*omega - gamma**2 - alpha**2*beta**2*omega \
    - alpha*beta*delta*gamma - beta*delta**2
print(f"\n(HL) : Delta = {sp.expand(Delta_man)}")
print(f"  residu  4*m3 - Delta  (doit etre 0) : {sp.simplify(4*m3_1 - Delta_man)}")

print("\n"+"="*72)
print("7. EIGENVALUES E0, E1, E2 (vérification)")
print("="*72)
J0 = J.subs({N:0, T:0, H:0})
print("E0 :", sp.simplify(J0.eigenvals()))

J1 = J.subs({N:1, T:0, H:0})
print("E1 :", sp.simplify(J1.eigenvals()))

J2 = J.subs({N:0, T:1, H: gamma/omega})
print("E2 :", sp.simplify(J2.eigenvals()))

print("\n"+"="*72)
print("8. TRANSCRITIQUE δ = δc : conditions de Sotomayor")
print("="*72)
# Au point δ = δc = ω/γ, E2 et E3 coïncident (N*=0)
delta_c = omega/gamma
J2_at_dc = J2.subs(delta, delta_c)
print("J(E2) à δ=δc :")
print(J2_at_dc)
print("Eigenvalues :", J2_at_dc.eigenvals())
# Une valeur propre nulle attendue (axe propre: direction transcritique)
# Vecteur propre à droite v et à gauche w pour λ=0
nullspace_R = J2_at_dc.nullspace()
nullspace_L = (J2_at_dc.T).nullspace()
print("Vecteur propre droit :", nullspace_R)
print("Vecteur propre gauche :", nullspace_L)

# Condition de Sotomayor pour transcritique : w^T F_delta(E2; δc) = 0
F_delta = sp.diff(F, delta)
F_delta_at = F_delta.subs({N:0, T:1, H: gamma/omega, delta: delta_c})
print("F_δ(E2; δc) =", F_delta_at.T)

if nullspace_L:
    w = nullspace_L[0]
    cond1 = sp.simplify((w.T * F_delta_at)[0])
    print("Sotomayor (1) w^T F_δ =", cond1)

if nullspace_R and nullspace_L:
    v = nullspace_R[0]
    w = nullspace_L[0]
    # Condition (2) : w^T D F_δ v
    DF_delta = F_delta.jacobian([N, T, H]).subs({N:0, T:1, H: gamma/omega, delta: delta_c})
    cond2 = sp.simplify((w.T * DF_delta * v)[0])
    print("Sotomayor (2) w^T (DF_δ) v =", cond2)

    # Condition (3) : w^T D^2 F (v, v)
    # D^2 F (v, v) coordonnée par coordonnée
    D2Fv = sp.zeros(3, 1)
    vars_ = [N, T, H]
    for i in range(3):
        Hess_i = sp.hessian(F[i], vars_).subs({N:0, T:1, H: gamma/omega, delta: delta_c})
        D2Fv[i] = (v.T * Hess_i * v)[0]
    cond3 = sp.simplify((w.T * D2Fv)[0])
    print("Sotomayor (3) w^T D^2F(v,v) =", cond3)

    # Invariance : ni cond2 ni cond3 ne sont invariants sous v -> c v
    # (cond2 est lineaire en v, cond3 quadratique), pas davantage le
    # quotient cond3/(w.v).  Ce qui l'est, c'est la pente de la branche
    # bifurquee, -2 sigma1 v / sigma2, dont la premiere composante doit
    # redonner dN*/ddelta en delta = delta_c.
    c_scale = sp.symbols('c', positive=True)
    quot = sp.simplify(cond3 / (w.T * v)[0])
    quot_scaled = sp.simplify((c_scale**2 * cond3) / ((w.T * (c_scale * v))[0]))
    print("  quotient (1/2) w^T D2F(v,v)/(w^T v) sous v -> c v : facteur",
          sp.simplify(quot_scaled / quot), "(donc non invariant)")
    slope = sp.simplify(-2 * cond2 * v / cond3)
    N_star = (omega - delta*gamma) / (omega - alpha*delta*gamma)
    dN = sp.simplify(sp.diff(N_star, delta).subs(delta, delta_c))
    print("  pente -2 sigma1 v/sigma2, composante N :", sp.simplify(slope[0]))
    print("  dN*/ddelta en delta_c                 :", dN)
    print("  residu (doit etre 0)                  :",
          sp.simplify(slope[0] - dN))

    # Meme calcul sur l'autre courbe transcritique, Gamma_alpha, en E1 :
    # le manuscrit annonce ces trois coefficients comme verifies
    # symboliquement, ce qui exige de deriver par alpha et non par delta.
    print("\n  Transcritique alpha = 1 (Gamma_alpha), en E1 = (1,0,0) :")
    J1_at = J.subs({N: 1, T: 0, H: 0, alpha: 1})
    v1 = J1_at.nullspace()[0]
    w1 = (J1_at.T).nullspace()[0]
    F_alpha = sp.diff(F, alpha)
    F_alpha_at = F_alpha.subs({N: 1, T: 0, H: 0, alpha: 1})
    DF_alpha = F_alpha.jacobian([N, T, H]).subs({N: 1, T: 0, H: 0, alpha: 1})
    D2Fv1 = sp.Matrix([
        (v1.T * sp.hessian(F[i], [N, T, H]).subs({N: 1, T: 0, H: 0, alpha: 1}) * v1)[0]
        for i in range(3)])
    print("    w^T F_alpha        =", sp.simplify((w1.T * F_alpha_at)[0]),
          "  (doit etre 0)")
    print("    w^T (DF_alpha) v   =", sp.simplify((w1.T * DF_alpha * v1)[0]))
    print("    w^T D^2F(v,v)      =", sp.simplify((w1.T * D2Fv1)[0]))
    print("    v =", v1.T, "  w =", w1.T)


# ──────────────────────────────────────────────────────────────────────
# Spatial linear stability at E_3 — coefficients of the perturbed
# characteristic polynomial det(J(E_3) - mu D - lam I) = 0.
# Used in Proposition prop:spatial_stab to show that no Turing
# instability can arise.
# ──────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("6bis. SPATIAL STABILITY (Proposition spatial_stab, case E_3)")
print("="*72)

mu = sp.symbols('mu', nonnegative=True)
lam = sp.symbols('lambda')

# Simplified Jacobian at E_3 using equilibrium relations
J3_simpl = sp.Matrix([
    [-N,        0,        -delta*N],
    [-alpha*beta*T, -beta*T, 0],
    [0,        gamma,    -omega]
])
D_diag = sp.diag(delta1, delta2, 1)
J3k = J3_simpl - mu * D_diag

chi = sp.expand(-(J3k - lam * sp.eye(3)).det())
poly = sp.Poly(chi, lam)
a1_t, a2_t, a3_t = [poly.all_coeffs()[i] for i in (1, 2, 3)]
a1_0, a2_0, a3_0 = (a1_t.subs(mu, 0), a2_t.subs(mu, 0), a3_t.subs(mu, 0))

print("a1_tilde(mu) =", sp.collect(sp.expand(a1_t), mu))
print("a2_tilde(mu) =", sp.collect(sp.expand(a2_t), mu))
print("a3_tilde(mu) =", sp.collect(sp.expand(a3_t), mu))
print()
print("a1(0)        =", sp.expand(a1_0))
print("a2(0)        =", sp.expand(a2_0))
print("a3(0)        =", sp.expand(a3_0))

print()
print("Differences (a_i_tilde - a_i)  (each must have non-negative coefficients):")
for name, da in (("a1_tilde - a1", sp.expand(a1_t - a1_0)),
                 ("a2_tilde - a2", sp.expand(a2_t - a2_0)),
                 ("a3_tilde - a3", sp.expand(a3_t - a3_0))):
    print(f"  {name} = {sp.collect(da, mu)}")

# Routh-Hurwitz margin and its difference, to confirm
# tilde a_1 tilde a_2 - tilde a_3 >= a_1 a_2 - a_3 for all mu >= 0
margin_tilde = sp.expand(a1_t * a2_t - a3_t)
margin_0 = sp.expand(a1_0 * a2_0 - a3_0)
diff_margin = sp.expand(margin_tilde - margin_0)
print()
print("Routh-Hurwitz margin at mu = 0 (= a_1 a_2 - a_3):")
print(" ", margin_0)
print()
print("Difference (margin_tilde - margin_0), collected in mu")
print("(every coefficient of mu^k must be a sum of products of")
print(" positive symbols, hence non-negative):")
print(" ", sp.collect(diff_margin, mu))



# ──────────────────────────────────────────────────────────────────────
# 9. Closed-form Routh–Hurwitz identity  (Prop. spatial_stab, eq. RH_margin_mu)
#
#    With  A = N* + delta1 mu,  B = beta T* + delta2 mu,  C = omega + mu :
#        a1(mu) = A + B + C
#        a2(mu) = AB + BC + CA
#        a3(mu) = ABC - alpha beta delta gamma N* T*
#        a1 a2 - a3 = (A+B)(B+C)(C+A) + alpha beta delta gamma N* T*
#
#    The mu = 0 case is eq. (hopf_margin): the absence of Hopf bifurcation
#    is the zero-mode instance of the absence of Turing instability.
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("9. CLOSED-FORM RH IDENTITY (eq. RH_coeffs_mu / RH_margin_mu)")
print("=" * 72)

A = N + delta1 * mu
B = beta * T + delta2 * mu
C = omega + mu
cyc = alpha * beta * delta * gamma * N * T          # weight of the 3-cycle

checks = [
    ("a1_tilde  - (A+B+C)", a1_t - (A + B + C)),
    ("a2_tilde  - (AB+BC+CA)", a2_t - (A*B + B*C + C*A)),
    ("a3_tilde  - (ABC - cyc)", a3_t - (A*B*C - cyc)),
    ("margin    - [(A+B)(B+C)(C+A) + cyc]",
     a1_t * a2_t - a3_t - ((A+B)*(B+C)*(C+A) + cyc)),
]
for name, expr in checks:
    residual = sp.simplify(sp.expand(expr))
    status = "OK" if residual == 0 else "FAIL"
    print(f"  [{status}] {name} = {residual}")

# The two off-diagonal products all vanish: no 2-cycle in the interaction
# graph. This is the structural reason the identity holds, and it is what
# rules out Turing instability -- not the diagonality of D.
print("\n  2-cycle products J_ij * J_ji (all must vanish):")
for i, j in ((0, 1), (0, 2), (1, 2)):
    print(f"    J[{i}][{j}] * J[{j}][{i}] = "
          f"{sp.simplify(J3_simpl[i, j] * J3_simpl[j, i])}")


# ──────────────────────────────────────────────────────────────────────
# 10. Existence of E3 on TWO branches, and the saddle on branch (ii)
#     (Proposition prop:equilibria, Theorem thm:local_stability (iv)-(v))
#
#     p = omega - delta gamma,  q = omega - alpha delta gamma,
#     q - p = delta gamma (1 - alpha).
#     E3 admissible  <=>  sign p = sign q = sign(1 - alpha), i.e.
#       branch (i)  : alpha < 1  and  delta < delta_c   -> stable node/focus
#       branch (ii) : alpha > 1  and  delta > delta_c   -> saddle (a3 < 0)
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("10. TWO EXISTENCE BRANCHES OF E3 AND THE SADDLE (Prop. equilibria)")
print("=" * 72)

al, dl = sp.symbols('al dl', real=True)          # alpha, delta without positivity
p_ = omega - dl * gamma
q_ = omega - al * dl * gamma
print("  [OK] q - p - delta*gamma*(1-alpha) =",
      sp.simplify(q_ - p_ - dl * gamma * (1 - al)))

# Numerical scan over the (alpha, delta) plane with the manuscript's
# (beta, gamma, omega) = (1.5, 5, 8), so delta_c = 1.6.
import numpy as np

b_, g_, w_ = 1.5, 5.0, 8.0
dc_ = w_ / g_
print(f"\n  Scan with (beta, gamma, omega) = ({b_}, {g_}, {w_}), "
      f"delta_c = {dc_}")
print(f"  {'alpha':>6} {'delta':>6} | {'N*':>8} {'T*':>8} {'H*':>8} "
      f"| {'a3':>9} | {'Re lam_max':>10} | type")
print("  " + "-" * 74)
for al_, dl_ in [(0.3, 0.5), (0.9, 1.5), (1.2, 2.5), (2.0, 3.0),
                 (1.5, 1.7), (0.3, 2.5), (1.2, 0.5)]:
    q_num = w_ - al_ * dl_ * g_
    Nn = (w_ - dl_ * g_) / q_num
    Tn = w_ * (1 - al_) / q_num
    Hn = g_ * (1 - al_) / q_num
    if min(Nn, Tn, Hn) <= 0:
        print(f"  {al_:6.2f} {dl_:6.2f} | {'--':>8} {'--':>8} {'--':>8} "
              f"| {'--':>9} | {'--':>10} | E3 not admissible")
        continue
    Jn = np.array([[-Nn, 0, -dl_ * Nn],
                   [-al_ * b_ * Tn, -b_ * Tn, 0],
                   [0, g_, -w_]])
    a3n = b_ * Nn * Tn * q_num
    ev = np.linalg.eigvals(Jn)
    kind = "stable" if max(ev.real) < 0 else "SADDLE"
    print(f"  {al_:6.2f} {dl_:6.2f} | {Nn:8.4f} {Tn:8.4f} {Hn:8.4f} "
          f"| {a3n:9.4f} | {max(ev.real):10.4f} | {kind}")

print("\n  Branch (ii) (alpha > 1, delta > delta_c) yields a3 < 0, hence")
print("  exactly one eigenvalue in the right half-plane: E3 is a saddle")
print("  of the KINETICS, with a 2-D stable manifold.  For the PDE the")
print("  count is different -- see Section 12.")

print("\nDONE.")


# ──────────────────────────────────────────────────────────────────────
# 11. The codim-2 point P* = (1, delta_c): a SEGMENT of equilibria
#     (Proposition prop:continuum, and the slow flow eq:slow_flow)
#
#     At alpha = 1, delta = delta_c = omega/gamma, the reaction field
#     vanishes identically on
#         S = { (s, 1-s, gamma(1-s)/omega) : s in [0,1] },
#     which joins E2 (s=0) to E1 (s=1).  S is a normally hyperbolic
#     attracting 1-D manifold of equilibria, so the local unfolding of P*
#     is a SCALAR flow (Fenichel), not a 2-D centre-manifold normal form.
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("11. SEGMENT D'EQUILIBRES EN P* = (1, delta_c)  ET FLOT LENT")
print("=" * 72)

s = sp.symbols('s', real=True)
a_, d_ = sp.symbols('a d', real=True)          # alpha = 1 + a, delta = dc(1+d)
dc = omega / gamma

seg = {N: s, T: 1 - s, H: gamma * (1 - s) / omega}
F_on_S = sp.simplify(F.subs({alpha: 1, delta: dc}).subs(seg))
print("\n  F on S at P*  (must be 0) :", F_on_S.T)

J_gen = F.jacobian([N, T, H])
J_S = sp.simplify(J_gen.subs({alpha: 1, delta: dc}).subs(seg))
print("\n  Jacobian along S :")
sp.pprint(J_S)

tau = sp.Matrix([1, -1, -gamma / omega])       # tangent vector to S
print("\n  J(s) * tau  (must be 0) :", sp.simplify(J_S * tau).T)

lam = sp.symbols('lambda')
chi = sp.Poly(sp.expand(sp.det(lam * sp.eye(3) - J_S)), lam).all_coeffs()
A_s = sp.factor(sp.simplify(chi[1]))
B_s = sp.factor(sp.simplify(chi[2]))
print(f"\n  char. poly = lambda (lambda^2 + A lambda + B), constant term = {chi[3]}")
print(f"    A(s) = {A_s}")
print(f"    B(s) = {B_s}")
print(f"    A(s) - [s + beta(1-s) + omega]        = "
      f"{sp.simplify(A_s - (s + beta*(1-s) + omega))}")
print(f"    B(s) - [omega s + beta(1-s)(omega+s)] = "
      f"{sp.simplify(B_s - (omega*s + beta*(1-s)*(omega+s)))}")
print("    -> A, B > 0 on [0,1]: the zero eigenvalue is simple and the two")
print("       transverse eigenvalues have negative real part (S attracting).")

# Left null vector of J(s), used to project the perturbation on the tangent.
ell = sp.Matrix([[-beta * (1 - s), s, beta * s * (1 - s) / gamma]])
print("\n  ell(s) * J(s)  (must be 0) :", sp.simplify(ell * J_S))

# Unfolding of E3: alpha = 1 + a, delta = dc (1 + d).
unf = {alpha: 1 + a_, delta: dc * (1 + d_)}
q_unf = sp.simplify((omega - alpha * delta * gamma).subs(unf))
N_unf = sp.simplify(((omega - delta * gamma) / (omega - alpha * delta * gamma)).subs(unf))
T_unf = sp.simplify((omega * (1 - alpha) / (omega - alpha * delta * gamma)).subs(unf))
print(f"\n  q  = {sp.factor(q_unf)}")
print(f"  N* = {N_unf}          T* = {T_unf}")
print(f"  N* + T* - (a+d)/(a+d+ad) = "
      f"{sp.simplify(N_unf + T_unf - (a_ + d_)/(a_ + d_ + a_*d_))}   (-> 1)")

eps, k = sp.symbols('epsilon k', positive=True)
print(f"  limit of N* along (a,d) = eps*(1,k) : "
      f"{sp.limit(N_unf.subs({a_: eps, d_: k*eps}), eps, 0)}    (= k/(k+1))")

# Perturbed field restricted to S, and the reduced scalar flow.
F_pert = sp.simplify(F.subs(unf).subs(seg))
print(f"\n  F on S with (a,d) : {sp.factor(F_pert).T}")
print(f"    = -s(1-s) (d, a beta, 0)^T ?  residual = "
      f"{sp.simplify(F_pert + s*(1-s)*sp.Matrix([d_, a_*beta, 0]))}")

num = sp.simplify((ell * F_pert)[0, 0])
den = sp.simplify((ell * tau)[0, 0])
kappa = s + beta * (1 - s) + beta * s * (1 - s) / omega
print(f"\n  ell . tau  = {sp.factor(den)}   (= -kappa(s), residual "
      f"{sp.simplify(den + kappa)})")
sdot = sp.simplify(num / den)
sdot_manuscript = beta * s * (1 - s) * (a_ * s - d_ * (1 - s)) / kappa
print(f"  reduced flow  sdot = {sp.factor(sdot)}")
print(f"  residual against manuscript form = "
      f"{sp.simplify(sdot - sdot_manuscript)}")
root = sp.solve(sp.Eq(a_ * s - d_ * (1 - s), 0), s)[0]
print(f"  interior root of the reduced field : s* = {sp.simplify(root)} "
      f"  (= N* + O(|a|+|d|))")
print("  slope of the bracket in s = a + d : the interior root attracts for")
print("  a + d < 0 (branch (i)) and repels for a + d > 0 (branch (ii)).")


# Numerical control: integrate the full 3-D kinetics and the reduced scalar
# equation from the SAME initial datum on S, and compare the equilibrium they
# select.  On branch (i) both settle on the interior root, and the residual
# gap is the O(eps) displacement between s* = d/(a+d) and N* = d/(a+d+ad).
# On branch (ii) the interior root repels and both models leave for the same
# endpoint, which is what the comparison tests there.
import numpy as np
from scipy.integrate import solve_ivp

BN, GN, WN = 1.5, 5.0, 8.0                 # beta, gamma, omega
DC = WN / GN


def _full(t, y, al, de):
    n, tt, h = y
    return [n * (1 - n - de * h), BN * tt * (1 - tt - al * n), GN * tt - WN * h]


def _reduced(t, y, av, dv):
    sv = y[0]
    kap = sv + BN * (1 - sv) + BN * sv * (1 - sv) / WN
    return [BN * sv * (1 - sv) * (av * sv - dv * (1 - sv)) / kap]


print("\n  Numerical control (full 3-D kinetics vs reduced scalar flow),")
print("  initial datum on S at s0 = 0.5 :")
print(f"  {'eps':>6} {'(a0,d0)':>10} {'3-D limit':>12} {'reduced':>10} "
      f"{'N* exact':>10} {'s* reduced':>11}")
for _eps in (1e-2, 5e-2):
    for _a0, _d0 in ((1.0, 2.0), (-1.0, -2.0), (1.0, 0.5)):
        _av, _dv = _eps * _a0, _eps * _d0
        _al, _de = 1 + _av, DC * (1 + _dv)
        _q = WN - _al * _de * GN
        _Ns = (WN - _de * GN) / _q
        _s0 = 0.5
        _y0 = [_s0, 1 - _s0, GN * (1 - _s0) / WN]
        _f = solve_ivp(_full, [0, 600], _y0, args=(_al, _de),
                       rtol=1e-10, atol=1e-12)
        _r = solve_ivp(_reduced, [0, 600], [_s0], args=(_av, _dv),
                       rtol=1e-10, atol=1e-12)
        print(f"  {_eps:6.3f} ({_a0:+.1f},{_d0:+.1f}) {_f.y[0, -1]:12.6f} "
              f"{_r.y[0, -1]:10.6f} {_Ns:10.6f} {_dv/(_av+_dv):11.6f}")
print("  Both models select the same equilibrium in every case; on branch (i)")
print("  the residual gap matches s* - N* = a d^2 / [(a+d)(a+d+ad)].")


# ──────────────────────────────────────────────────────────────────────
# 12. Modal spectrum of E3 on branch (ii): dim W^u = 4, NOT 1
#     (Remark rem:separatrix_pde)
#
#     The kinetic saddle has a 2-D stable manifold in R^3, but as a steady
#     state of the PDE on Omega = (0,L)^2 with Neumann conditions the
#     unstable eigenvalue survives on every mode with mu_mn < mu*.  With the
#     parameters of Section 4.5 and L = 2, four modes are unstable, so
#     W^s(E3) has codimension 4 and cannot bound the two basins.
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("12. SPECTRE MODAL DE E3 SUR LA BRANCHE (ii)  ->  dim W^u = 4")
print("=" * 72)

import numpy as np
from scipy.optimize import brentq

AL, BE, GA, OM, DE = 1.2, 1.5, 5.0, 8.0, 2.5     # Section 4.5 parameters
D1 = D2 = 5e-3
L = 2.0

q_ii = OM - AL * DE * GA
E3 = np.array([(OM - DE * GA) / q_ii, OM * (1 - AL) / q_ii, GA * (1 - AL) / q_ii])
J3n = np.array([[-E3[0], 0.0, -DE * E3[0]],
                [-AL * BE * E3[1], -BE * E3[1], 0.0],
                [0.0, GA, -OM]])
Dm = np.diag([D1, D2, 1.0])
print(f"\n  E3 = ({E3[0]:.6f}, {E3[1]:.6f}, {E3[2]:.6f}),  q = {q_ii:.4f} < 0")
print(f"\n  {'(m,n)':>7} {'mu_mn':>9}   eigenvalues of J(E3) - mu D          unstable")
n_unstable = 0
for m in range(4):
    for n in range(4):
        mu = np.pi ** 2 * (m * m + n * n) / L ** 2
        ev = np.sort(np.linalg.eigvals(J3n - mu * Dm).real)
        bad = (ev > 0).sum()
        n_unstable += bad
        if m <= 2 and n <= 2:
            print(f"  ({m},{n})   {mu:9.4f}   {ev[0]:9.4f} {ev[1]:9.4f} {ev[2]:9.4f}"
                  f"      {'YES' if bad else 'no'}")
print(f"\n  unstable eigenvalues summed over Neumann modes : {n_unstable}"
      f"  ->  dim W^u(E3) = {n_unstable}, codim W^s(E3) = {n_unstable}")

mu_star = brentq(lambda mu: np.linalg.eigvals(J3n - mu * Dm).real.max(), 0.1, 50.0)
L_star = np.pi / np.sqrt(mu_star)
print(f"  critical modal threshold  mu* = {mu_star:.4f}")
print(f"  all non-constant modes damped iff  pi^2/L^2 > mu*,  i.e.  L < {L_star:.4f}")
print(f"  domain used in Section 4.5 : L = {L}  ->  mu_10 = "
      f"{np.pi**2/L**2:.4f} < mu*, so the constant saddle is unstable in 4 directions.")
print("\n  A manifold of codimension 4 cannot be the boundary between two basins,")
print("  so for GENERAL data the spatial threshold is carried by a non-constant")
print("  state.  Caveat: the centred data of Section 4.5 stay in the reflection-")
print("  symmetric subspace, from which the modes (1,0), (0,1), (1,1) are absent;")
print("  there the first non-constant mode is (2,0) with mu = 9.87 > mu*, so E3")
print("  keeps a 1-D unstable manifold and W^s(E3) DOES separate the basins.")

print("\nDONE.")

# ──────────────────────────────────────────────────────────────────────
# 13. Cooperative structure and the global trichotomy
#     (Proposition prop:monotone, Lemma lem:open_box, Theorem thm:trichotomy)
#
#     S J S is Metzler for S = diag(1,-1,-1), so the system is cooperative in
#     the variables (N,-T,-H).  The open order interval B = (0,1)^2 x (0,g/w)
#     is positively invariant, contains E3 and no other equilibrium, and the
#     order argument gives global convergence on the three non-bistable
#     regions -- with no hypothesis, in particular without (HL).
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("13. STRUCTURE COOPERATIVE ET TRICHOTOMIE GLOBALE")
print("=" * 72)

S_ = sp.diag(1, -1, -1)
M_ = sp.simplify(S_ * J_gen * S_)
print("\n  S J S, entrees hors-diagonale (doivent etre >= 0 sur R) :")
for i_ in range(3):
    for j_ in range(3):
        if i_ != j_:
            print(f"    ({i_+1},{j_+1}) = {sp.simplify(M_[i_, j_])}")

# Invariance de la boite ouverte : le manuscrit ne procede plus par faces
# (l'argument y degenerait sur les aretes) mais par six inegalites
# differentielles y' >= -c y sur les distances au bord de
# B_eta = (0,1)^2 x (0, gamma/omega + eta).  Chaque distance est introduite
# comme une variable a part entiere, et l'on verifie que y' + c y est une
# somme de termes positifs sur B_eta pour le c indique.
print("\n  Inegalites differentielles sur les distances au bord de B_eta :")
y = sp.symbols('y', positive=True)
eta = sp.symbols('eta', positive=True)
M0b = gamma / omega + eta
rows = [
    ("N",                    F[0].subs(N, y),               delta * M0b),
    ("1 - N",               -F[0].subs(N, 1 - y),           sp.Integer(1)),
    ("T",                    F[1].subs(T, y),               alpha * beta),
    ("1 - T",               -F[1].subs(T, 1 - y),           beta),
    ("H",                    F[2].subs(H, y),               omega),
    ("gamma/omega + eta - H", -F[2].subs(H, M0b - y),       omega),
]
for name, rate, c in rows:
    expr = sp.simplify(sp.expand(rate + c * y))
    print(f"    y = {name:>21} :  y' + ({str(c)}) y  =  {expr}")
print("    -> chaque membre de droite est une somme de termes positifs sur")
print("       B_eta (0 <= N, T <= 1, 0 <= H <= gamma/omega + eta), donc")
print("       y' >= -c y et y reste > 0 si elle part > 0.")

# Global convergence, numerically, over random parameters and initial data.
def _kin(t, y, prm):
    a_, b_, g_, w_, d_ = prm
    n_, t_, h_ = y
    return [n_ * (1 - n_ - d_ * h_), b_ * t_ * (1 - t_ - a_ * n_), g_ * t_ - w_ * h_]


def _limit(prm, y0, tf=60000.0):
    return solve_ivp(_kin, [0, tf], y0, args=(prm,),
                     rtol=1e-11, atol=1e-13, method="LSODA").y[:, -1]


_rng = np.random.default_rng(20260801)
print("\n  Convergence globale, 120 jeux de parametres x 8 donnees initiales :")
print(f"  {'region':>34}  {'(HL) echoue':>12}  {'ecart max':>11}")
for _label, _pick in (
        ("(i)   alpha<1, delta<dc  -> E3", "i"),
        ("(ii)  alpha<1, delta>dc  -> E2", "ii"),
        ("(iii) alpha>1, delta<dc  -> E1", "iii")):
    _worst, _nHL = 0.0, 0
    for _ in range(120):
        _b = 10 ** _rng.uniform(-1, 1)
        _g = 10 ** _rng.uniform(-1, 1.3)
        _w = 10 ** _rng.uniform(-1, 1.3)
        _dc = _w / _g
        if _pick == "i":
            _a = _rng.uniform(0.02, 0.999); _d = _rng.uniform(1e-3, _dc * 0.999)
            _q = _w - _a * _d * _g
            _target = np.array([(_w - _d * _g) / _q, _w * (1 - _a) / _q, _g * (1 - _a) / _q])
        elif _pick == "ii":
            _a = _rng.uniform(0.02, 0.999); _d = _dc * _rng.uniform(1.001, 12.0)
            _target = np.array([0.0, 1.0, _g / _w])
        else:
            _a = _rng.uniform(1.001, 5.0); _d = _rng.uniform(1e-3, _dc * 0.999)
            _target = np.array([1.0, 0.0, 0.0])
        _Delta = (4 * _b * _w - _g ** 2 - _a ** 2 * _b ** 2 * _w
                  - _a * _b * _d * _g - _b * _d ** 2)
        _nHL += (_Delta <= 0) or (_a ** 2 * _b >= 4.0)   # (HL) = conjonction
        for _ in range(8):
            _y0 = [_rng.uniform(1e-3, 1), _rng.uniform(1e-3, 1), _rng.uniform(0, _g / _w)]
            _worst = max(_worst, np.abs(_limit((_a, _b, _g, _w, _d), _y0) - _target).max())
    print(f"  {_label:>34}  {_nHL:>7}/120  {_worst:>11.2e}")
print("  -> la convergence ne depend pas de (HL), qui echoue sur une large part")
print("     des jeux de la branche (i) sans que rien ne change.")

# ──────────────────────────────────────────────────────────────────────
# 14. Sharpness of the Volterra criterion (Remark rem:weights)
#
#     With c1 = 1, the scale-invariant quantity is det Q(c)/(c1 c2 c3).  The
#     product of its three negative terms is independent of c, so AM-GM gives
#     the maximum in closed form, and 4 - s^3 - 3 s^2 = -(s-1)(s+2)^2 with
#     s = rho^(1/3) turns positivity into rho < 1, i.e. into q > 0.  The
#     Volterra family therefore certifies global stability exactly where E3 is
#     locally stable, and nowhere else.
# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("14. NETTETE DU CRITERE DE VOLTERRA")
print("=" * 72)

c2, c3, rho, sroot = sp.symbols('c2 c3 rho s', positive=True)
Qc = sp.Matrix([[1, c2 * alpha * beta / 2, delta / 2],
                [c2 * alpha * beta / 2, c2 * beta, -c3 * gamma / 2],
                [delta / 2, -c3 * gamma / 2, c3 * omega]])
Dq = sp.simplify(sp.det(Qc) / (c2 * c3))
print("\n  4 det Q(c)/(c2 c3) =", sp.expand(4 * Dq))

# The three c-dependent terms and their c-free product.
t1, t2, t3 = c3 * gamma**2 / c2, c2 * alpha**2 * beta**2 * omega, beta * delta**2 / c3
print("  produit des trois termes (doit etre sans c) :",
      sp.simplify(t1 * t2 * t3))

c2opt = rho**sp.Rational(2, 3) / (alpha**2 * beta)
c3opt = delta**2 / (omega * rho**sp.Rational(2, 3))
rho_def = alpha * delta * gamma / omega
Dopt = sp.simplify(Dq.subs({c2: c2opt, c3: c3opt}).subs(rho, rho_def))
Dclosed = beta * omega * (1 - rho_def / 4 - sp.Rational(3, 4) * rho_def**sp.Rational(2, 3))
print("  residu  D(c_opt) - beta*omega*(1 - rho/4 - (3/4) rho^(2/3)) =",
      sp.simplify(Dopt - Dclosed))
print("  factorisation  4 - s^3 - 3 s^2 + (s-1)(s+2)^2 =",
      sp.expand(4 - sroot**3 - 3 * sroot**2 + (sroot - 1) * (sroot + 2)**2))
print("  -> D(c_opt) > 0  <=>  s < 1  <=>  rho < 1  <=>  q = omega - alpha delta gamma > 0")

grad = [sp.simplify(sp.diff(Dq, v).subs({c2: c2opt, c3: c3opt}).subs(rho, rho_def))
        for v in (c2, c3)]
print(f"  gradient de D en c_opt : {grad}   (point stationnaire)")

_rng14 = np.random.default_rng(20260802)
_bad, _nocert = 0, 0
for _ in range(20000):
    _a, _b = _rng14.uniform(0.02, 5), 10 ** _rng14.uniform(-1, 1)
    _g, _w = 10 ** _rng14.uniform(-1, 1.3), 10 ** _rng14.uniform(-1, 1.3)
    _d = 10 ** _rng14.uniform(-2, 1.3)
    _rho = _a * _d * _g / _w
    _c2 = _rho ** (2 / 3) / (_a ** 2 * _b)
    _c3 = _d ** 2 / (_w * _rho ** (2 / 3))
    _Q = np.array([[1, _c2 * _a * _b / 2, _d / 2],
                   [_c2 * _a * _b / 2, _c2 * _b, -_c3 * _g / 2],
                   [_d / 2, -_c3 * _g / 2, _c3 * _w]])
    _pd = np.linalg.eigvalsh(_Q).min() > 0
    if _pd != (_rho < 1):
        _bad += 1
    if _rho >= 1:                      # aucun poids ne doit certifier
        for _ in range(20):
            _q2, _q3 = 10 ** _rng14.uniform(-4, 4), 10 ** _rng14.uniform(-4, 4)
            _Qr = np.array([[1, _q2 * _a * _b / 2, _d / 2],
                            [_q2 * _a * _b / 2, _q2 * _b, -_q3 * _g / 2],
                            [_d / 2, -_q3 * _g / 2, _q3 * _w]])
            if np.linalg.eigvalsh(_Qr).min() > 0:
                _nocert += 1
                break
print(f"\n  20000 tirages : desaccords entre 'Q(c_opt) > 0' et 'rho < 1' : {_bad}")
print(f"  certificats trouves par recherche aleatoire quand rho >= 1 : {_nocert}")



print("\n" + "=" * 72)
print("15. PROFIL QUASI-STATIQUE DE L'ACIDE (equation (bessel) du manuscrit)")
print("=" * 72)
print("""
  Le seuil en taille de la region bistable est attribue a la longueur
  d'ecrantage ell_H = omega^(-1/2), via le profil quasi-statique d'une source
  en disque de rayon R et de densite T0 dans le plan.  Le manuscrit ecrit ce
  profil et en tire R_c sans fitter de constante ; on verifie ici que le
  profil annonce resout bien le probleme, qu'il est C^1 a la traversee du
  bord du disque, et que sa valeur au centre est celle qui figure dans le
  texte.  Le calcul est fait en coordonnees radiales, ou le laplacien est
  u'' + u'/r.
""")

r_, R_, s_ = sp.symbols('r R s', positive=True)
gamma_, omega_, T0_ = sp.symbols('gamma omega T0', positive=True)

# s = sqrt(omega) ; les fonctions modifiees de Bessel I0, I1, K0, K1.
H_in = (gamma_ * T0_ / omega_) * (1 - s_ * R_ * sp.besselk(1, s_ * R_)
                                  * sp.besseli(0, s_ * r_))
H_out = (gamma_ * T0_ / omega_) * s_ * R_ * sp.besseli(1, s_ * R_) \
    * sp.besselk(0, s_ * r_)


def _radial_helmholtz(expr):
    """-Delta u + omega u en radial, avec omega = s^2."""
    lap = sp.diff(expr, r_, 2) + sp.diff(expr, r_) / r_
    return sp.simplify((-lap + omega_ * expr).subs(omega_, s_**2))


src_in = sp.simplify(_radial_helmholtz(H_in) - (gamma_ * T0_).subs(omega_, s_**2))
print("  -Delta H + omega H - gamma T0  a l'interieur (doit etre 0) :", src_in)
print("  -Delta H + omega H             a l'exterieur (doit etre 0) :",
      _radial_helmholtz(H_out))

saut = sp.simplify((H_in - H_out).subs(r_, R_))
saut_prime = sp.simplify((sp.diff(H_in, r_) - sp.diff(H_out, r_)).subs(r_, R_))
z_ = sp.symbols('z', positive=True)
saut_z = sp.simplify(saut.subs(R_, z_ / s_))
print("\n  saut de H   au bord du disque :", saut_z)
print("     -> vaut (gamma T0/omega) [1 - z (I0(z) K1(z) + I1(z) K0(z))], nul")
print("        par le wronskien, que sympy ne reduit pas seul et qu'on")
print("        controle numeriquement ci-dessous.")
print("  saut de H'  au bord du disque (doit etre 0) :", saut_prime)

centre = sp.simplify(H_in.subs(r_, 0))
print("\n  H(0) =", centre)
print("  == (gamma T0/omega)[1 - sqrt(omega) R K1(sqrt(omega) R)] :",
      sp.simplify(centre - (gamma_ * T0_ / omega_)
                  * (1 - s_ * R_ * sp.besselk(1, s_ * R_))))

# Controle numerique du wronskien et de la valeur R_c = 1.81 ell_H citee.
import numpy as _np15
from scipy.special import i0 as _i0, i1 as _i1, k0 as _k0, k1 as _k1
_z = _np15.array([0.05, 0.3, 1.0, 1.81, 5.0, 20.0])
_wr = _i0(_z) * _k1(_z) + _i1(_z) * _k0(_z) - 1.0 / _z
print(f"\n  wronskien I0 K1 + I1 K0 - 1/z, ecart max : {_np15.abs(_wr).max():.2e}")

_alpha15, _beta15, _gamma15, _omega15, _delta15 = 1.2, 1.5, 5.0, 8.0, 2.5
_rho15, _T015 = _delta15 * _gamma15 / _omega15, 0.95
_s15 = _np15.sqrt(_omega15)
_f = lambda R: _rho15 * _T015 * (1 - _s15 * R * _k1(_s15 * R)) - 1.0
_lo, _hi = 1e-6, 10.0
for _ in range(200):
    _mid = 0.5 * (_lo + _hi)
    _lo, _hi = (_lo, _mid) if _f(_mid) > 0 else (_mid, _hi)
_Rc15 = 0.5 * (_lo + _hi)
_ell15 = _omega15 ** -0.5
print(f"  delta H(0) = 1  =>  R_c = {_Rc15:.4f} = {_Rc15/_ell15:.3f} ell_H "
      f"(ell_H = {_ell15:.4f})")
print("  -> la condition ne fait intervenir omega que par ell_H et le rapport")
print("     invariant de jauge rho = delta gamma/omega.")
print("\nDONE.")


print("\n" + "=" * 72)
print("16. NON-TURING PAR COOPERATIVITE (raison structurelle)")
print("=" * 72)
print("""
  Le manuscrit tire l'absence d'instabilite de Turing d'une identite fermee
  de Routh-Hurwitz.  Cette identite reste utile, car elle donne le seuil
  mu*, mais la raison structurelle est plus courte : apres conjugaison par
  Sigma = diag(1,-1,-1) la jacobienne est de Metzler, et retrancher mu*D
  avec D diagonale positive la diminue entree par entree, donc diminue sa
  borne spectrale.  On verifie ici les deux faits sur le modele ETENDU par
  un terme de competition symetrique -alpha' N T, celui dont le manuscrit
  affirmait a tort qu'il rouvrait la question.
""")
_rng16 = np.random.default_rng(20260808)
_worst, _n16, _nonmetzler = -np.inf, 0, 0
for _ in range(20000):
    _a, _ap = _rng16.uniform(0.01, 3), _rng16.uniform(0, 3)
    _b = 10 ** _rng16.uniform(-1, 1)
    _g = 10 ** _rng16.uniform(-1, 1.3)
    _w = 10 ** _rng16.uniform(-1, 1.3)
    _d = 10 ** _rng16.uniform(-2, 1.3)
    _N, _T, _H = _rng16.uniform(0, 1), _rng16.uniform(0, 1), _rng16.uniform(0, _g / _w)
    _J = np.array([[1 - 2*_N - _ap*_T - _d*_H, -_ap*_N, -_d*_N],
                    [-_a*_b*_T, _b*(1 - 2*_T - _a*_N), 0.0],
                    [0.0, _g, -_w]])
    _S = np.diag([1., -1., -1.])
    _M = _S @ _J @ _S
    _off = _M - np.diag(np.diag(_M))
    if _off.min() < -1e-12:
        _nonmetzler += 1
        continue
    _D = np.diag([10 ** _rng16.uniform(-3, 0), 10 ** _rng16.uniform(-3, 0), 1.0])
    _s0 = max(np.linalg.eigvals(_J).real)
    _n16 += 1
    for _mu in (0.1, 1.0, 5.0, 20.0, 100.0):
        _worst = max(_worst, max(np.linalg.eigvals(_J - _mu*_D).real) - _s0)
print(f"  jacobiennes du modele symetrise testees : {_n16}")
print(f"  jacobiennes non-Metzler apres conjugaison : {_nonmetzler}  (attendu 0)")
print(f"  max de  s(J - mu D) - s(J)  sur {_n16} tirages x 5 modes : {_worst:.3e}")
print("  -> negatif partout : la stabilite cinetique se propage a tous les modes,")
print("     sans hypothese sur les cycles, et le terme symetrique ne rouvre rien.")
print("\nDONE.")
