"""Generate the manuscript figures from the verification data.

Run from the project root:  python3 Figures/make_figures.py
Requires matplotlib; all numbers are recomputed here, none are hard-coded.
"""
import math, random, sys, os
from fractions import Fraction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from weighted_sampler import ExactWeightedSampler

OUT = os.path.dirname(os.path.abspath(__file__))
INK, ACC, MUT = "#1a1a2e", "#2b6cb0", "#a0aec0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.dpi": 200})

# ---------------------------------------------------------------- Figure 1
def exact_expected_bits(n):
    r, k, tot, seen = 1 % n, 0, Fraction(0), {}
    while r and r not in seen:
        seen[r] = (k, tot); tot += Fraction(r, 2**k); r = (2*r) % n; k += 1
    if not r: return tot
    s0, pre = seen[r]
    return pre + (tot - pre) / (1 - Fraction(1, 2**(k-s0)))

ns = [3, 6, 10, 100, 1000]
base = [float(Fraction((n-1).bit_length() * 2**((n-1).bit_length()), n)) for n in ns]
cand = [float(exact_expected_bits(n)) for n in ns]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = range(len(ns)); w = 0.38
ax.bar([i-w/2 for i in x], base, w, label="Restart baseline", color=MUT)
ax.bar([i+w/2 for i in x], cand, w, label="Retained state", color=ACC)
ax.set_xticks(list(x)); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("Number of outcomes $n$"); ax.set_ylabel("Expected fair bits")
ax.legend(frameon=False, fontsize=7); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_caseA_bits.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
def entropy(w):
    W = sum(w)
    return -sum((x/W)*math.log2(x/W) for x in w if x > 0)

random.seed(4); Hs, Es = [], []
for _ in range(400):
    n = random.randint(2, 8)
    w = [random.randint(0, 10**random.randint(1, 6)) for _ in range(n)]
    if sum(w) == 0 or sum(1 for x in w if x > 0) < 2: continue
    s = ExactWeightedSampler(w)
    Hs.append(entropy(w)); Es.append(float(s.expected_bits_upper(depth=400)))
fig, ax = plt.subplots(figsize=(3.4, 2.6))
lo, hi = 0, max(Hs)*1.05
ax.fill_between([lo, hi], [lo, hi], [lo+2, hi+2], color=ACC, alpha=0.10,
                label="Admissible band $H\\leq E\\leq H+2$")
ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, label="$E=H$")
ax.plot([lo, hi], [lo+2, hi+2], color=INK, lw=0.8, ls="--", label="$E=H+2$")
ax.scatter(Hs, Es, s=5, color=ACC, alpha=0.65, edgecolors="none", zorder=3)
ax.set_xlabel("Shannon entropy $H$ (bits)"); ax.set_ylabel("Expected bits $E$")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_caseB_entropy.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
fig, ax = plt.subplots(figsize=(3.4, 2.2))
random.seed(11); shown = 0
for _ in range(400):
    if shown >= 40: break
    n = random.randint(3, 7)
    w = [random.randint(1, 10**random.randint(1, 8)) for _ in range(n)]
    s = ExactWeightedSampler(w)
    if s.degenerate is not None: continue
    ks = list(range(1, 25)); ms = [s._level(k)[1] for k in ks]
    ax.plot(ks, [m/(n-1) for m in ms], color=ACC, alpha=0.30, lw=0.7)
    shown += 1
ax.axhline(1.0, color=INK, lw=1.0, ls="--", label="Bound $m_k=n-1$")
ax.set_ylim(0, 1.15); ax.set_xlabel("Tree level $k$")
ax.set_ylabel("$m_k/(n-1)$")
ax.legend(frameon=False, fontsize=7); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_caseB_width.pdf")); plt.close(fig)
print("figures written to", OUT)
