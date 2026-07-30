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

# Test de la borne du manuscrit : alpha gamma delta < (2-delta)(2-alpha)(2 omega - gamma)
HL_man = (2 - delta)*(2 - alpha)*(2*omega - gamma) - alpha*gamma*delta
print(f"\n(HL) du manuscrit (>0 attendu) : {sp.expand(HL_man)}")

# Comparaison avec ce que donne la décomposition Young avec c1=c2=c3=1
# du manuscrit : delta < 2 - alpha*beta, alpha < 2 - gamma/beta, delta + gamma < 2 omega
# Or le manuscrit ÉCRIT (HL): alpha gamma delta < (2-delta)(2-alpha)(2 omega - gamma)
# Ce qui est INDÉPENDANT de beta -- contradiction avec le calcul Young.

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


# ──────────────────────────────────────────────────────────────────────
# Spatial linear stability at E_3 — coefficients of the perturbed
# characteristic polynomial det(J(E_3) - mu D - lam I) = 0.
# Used in Proposition prop:spatial_stab to show that no Turing
# instability can arise.
# ──────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("6. SPATIAL STABILITY (Proposition spatial_stab, case E_3)")
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
print("  whose 2D stable manifold separates the basins of E1 and E2.")

print("\nDONE.")
