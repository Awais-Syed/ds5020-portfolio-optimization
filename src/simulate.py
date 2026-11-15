"""Simulation check with a known truth.

Run from the project root:
    python src/simulate.py

Four assets with a chosen mu and Sigma are simulated, then every derivation is
tested numerically. Uses only numpy, pandas and matplotlib.
Each check says which derivation it confirms (D1..D5 = Q1..Q5 in the report).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
DOCS = ROOT / "docs"


# ---------------------------------------------------------------- the method
def abc(S, m):
    """Solve S u = 1 and S v = m (no explicit inverse); return A, B, C, Delta, u, v."""
    one = np.ones(len(m))
    u = np.linalg.solve(S, one)
    v = np.linalg.solve(S, m)
    A, B, C = one @ u, m @ u, m @ v
    return A, B, C, A * C - B ** 2, u, v


def gmv_weights(S, m):
    """Derivation 3: w* = S^{-1} 1 / (1' S^{-1} 1)."""
    A, B, C, D, u, v = abc(S, m)
    return u / A


def target_weights(S, m, r):
    """Derivation 4: solve the 2x2 system for (lambda, gamma), then w = S^{-1}(lambda m + gamma 1)."""
    A, B, C, D, u, v = abc(S, m)
    lam, gam = np.linalg.solve([[C, B], [B, A]], [r, 1.0])
    return lam * v + gam * u


def frontier_variance(S, m, r):
    """Derivation 5: sigma^2(r) = (A r^2 - 2 B r + C) / Delta."""
    A, B, C, D, u, v = abc(S, m)
    return (A * r ** 2 - 2 * B * r + C) / D


def flag(ok):
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------- the truth
rng = np.random.default_rng(0)
mu = np.array([0.05, 0.08, 0.12, 0.07])
vol = np.array([0.10, 0.15, 0.25, 0.12])
Corr = np.array([[1, .3, .2, .1],
                 [.3, 1, .4, .2],
                 [.2, .4, 1, .3],
                 [.1, .2, .3, 1]])
Sigma = np.outer(vol, vol) * Corr
n = len(mu)
one = np.ones(n)
T_BIG, T_SMALL = 2000, 60

R = rng.multivariate_normal(mu, Sigma, size=T_BIG)
S = np.cov(R.T)
m = R.mean(axis=0)

print("=== SIMULATION SETUP ===")
print("Model       : R ~ N(mu, Sigma), n = 4 assets, seed 0")
print("True mu     :", mu)
print("True vol    :", vol)
print("Sample size :", T_BIG, "(and", T_SMALL, "for the small-sample comparison)")

# ------------------------------------------------------------ check 1 (D2)
print("\n=== CHECK 1: Sigma-hat is symmetric PSD (Derivation 2) ===")
eig = np.linalg.eigvalsh(S)
print("eigenvalues of Sigma-hat:", np.round(eig, 6))
print("symmetric:", np.allclose(S, S.T), "| min eigenvalue >= -1e-12:", flag(eig.min() >= -1e-12))
print("true Sigma eigenvalues  :", np.round(np.linalg.eigvalsh(Sigma), 6))

# ------------------------------------------------------------ check 2 (D1)
print("\n=== CHECK 2: portfolio mean and variance (Derivation 1) ===")
w_min = gmv_weights(S, m)
port = R @ w_min
var_formula = w_min @ S @ w_min
var_sample = port.var(ddof=1)
print("w_min                    :", np.round(w_min, 4), "| sum =", round(w_min.sum(), 12))
print(f"w' S w                   : {var_formula:.10f}")
print(f"sample var of R @ w      : {var_sample:.10f}")
print(f"mean of R @ w vs w' m    : {port.mean():.10f} vs {w_min @ m:.10f}")
print("variance identity:", flag(np.isclose(var_formula, var_sample, rtol=1e-10)),
      "| mean identity:", flag(np.isclose(port.mean(), w_min @ m, rtol=1e-10)))

# ------------------------------------------------------------ check 3 (D3)
print("\n=== CHECK 3: no weight vector with sum 1 beats w* (Derivation 3) ===")
A, B, C, D, u, v = abc(S, m)
N = 200_000
Wr = rng.normal(0, 1, size=(N, n))
Wr[:, -1] = 1 - Wr[:, :-1].sum(axis=1)          # forces 1'w = 1 exactly
var_rand = np.einsum("ij,jk,ik->i", Wr, S, Wr)   # w' S w for every row
d = rng.normal(0, 0.01, size=(N, n))
d -= d.mean(axis=1, keepdims=True)               # 1'd = 0, so w* + d is feasible
Wn = w_min + d
var_near = np.einsum("ij,jk,ik->i", Wn, S, Wn)
print(f"1/A (variance at w*)           : {1 / A:.8f}  (= w*' S w* = {var_formula:.8f})")
print(f"smallest of {N} random weights : {var_rand.min():.8f}")
print(f"smallest of {N} nearby weights : {var_near.min():.8f}")
print("none beats w*:", flag(var_rand.min() >= var_formula - 1e-12 and var_near.min() >= var_formula - 1e-12))
print("variance at w* equals 1/A:", flag(np.isclose(var_formula, 1 / A, rtol=1e-10)))

# --------------------------------------------------------- check 4/5 (D4, D5)
print("\n=== CHECK 4/5: 2x2 system and the frontier parabola (Derivations 4, 5) ===")
rows = []
for r in np.linspace(0.04, 0.14, 6):
    w = target_weights(S, m, r)
    rows.append({
        "r": r,
        "sum(w)": w.sum(),
        "m'w": m @ w,
        "w'Sw": w @ S @ w,
        "(A r^2-2Br+C)/Delta": frontier_variance(S, m, r),
    })
tab = pd.DataFrame(rows)
print(tab.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
print("constraints hold:", flag(np.allclose(tab["sum(w)"], 1) and np.allclose(tab["m'w"], tab["r"])))
print("parabola matches w'Sw:", flag(np.allclose(tab["w'Sw"], tab["(A r^2-2Br+C)/Delta"], rtol=1e-9)))
print(f"vertex: sigma^2 = 1/A = {1 / A:.8f} at r = B/A = {B / A:.6f}")
w_vertex = target_weights(S, m, B / A)
print("w*(B/A) equals w_min:", flag(np.allclose(w_vertex, w_min)))

# ------------------------------------ estimation error: 2000 vs 60 vs truth
print("\n=== ESTIMATION ERROR: true parameters vs T=2000 vs T=60 ===")
R60 = R[:T_SMALL]
S60, m60 = np.cov(R60.T), R60.mean(axis=0)
sets = {"true": (Sigma, mu), "T=2000": (S, m), "T=60": (S60, m60)}

print("\nMean vector (true | T=2000 | T=60):")
print(pd.DataFrame({k: v_[1] for k, v_ in sets.items()}).round(4).to_string())
print("\nVolatilities (true | T=2000 | T=60):")
print(pd.DataFrame({k: np.sqrt(np.diag(v_[0])) for k, v_ in sets.items()}).round(4).to_string())

R_TARGET = 0.10
W = {}
for k, (Sk, mk) in sets.items():
    W[("GMV", k)] = gmv_weights(Sk, mk)
    W[(f"r={R_TARGET}", k)] = target_weights(Sk, mk, R_TARGET)
Wdf = pd.DataFrame(W, index=[f"asset {i + 1}" for i in range(n)])
print("\nWeights (columns: portfolio, estimate source):")
print(Wdf.round(4).to_string())

A_t, B_t, C_t, D_t, _, _ = abc(Sigma, mu)
print(f"\nTrue minimum variance 1/A = {1 / A_t:.6f} (vol {np.sqrt(1 / A_t):.4f}); true B/A = {B_t / A_t:.4f}")
print("True risk of each estimated GMV portfolio (w' Sigma_true w), and its excess over the true minimum:")
for k in sets:
    wk = W[("GMV", k)]
    rk = wk @ Sigma @ wk
    print(f"  {k:7s}: variance {rk:.6f}  vol {np.sqrt(rk):.4f}  excess {rk - 1 / A_t:.6f}")
print("Distance of estimated GMV weights from true GMV weights (max abs):")
for k in ("T=2000", "T=60"):
    print(f"  {k}: {np.abs(W[('GMV', k)] - W[('GMV', 'true')]).max():.4f}")
print("Distance of estimated r=0.10 weights from true r=0.10 weights (max abs):")
for k in ("T=2000", "T=60"):
    print(f"  {k}: {np.abs(W[(f'r={R_TARGET}', k)] - W[(f'r={R_TARGET}', 'true')]).max():.4f}")

# ------------------------------------------------------------------- figure
rs = np.linspace(0.03, 0.16, 200)
fig, ax = plt.subplots(figsize=(7, 4.5))
for k, (Sk, mk) in sets.items():
    ax.plot([frontier_variance(Sk, mk, r) for r in rs], rs, label=f"{k} parameters")
ax.set_xlabel(r"variance $\sigma^2$")
ax.set_ylabel("expected return r")
ax.set_title("Frontier parabola: true vs estimated (simulation)")
ax.legend()
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "simulation_frontier.png", dpi=150)
plt.close(fig)
print("\nSaved", FIGS / "simulation_frontier.png")

# save the comparison table for the report
DOCS.mkdir(exist_ok=True)
Wdf.round(4).to_csv(DOCS / "simulation_weights.csv")
print("Saved", DOCS / "simulation_weights.csv")