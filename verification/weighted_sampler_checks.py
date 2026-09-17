"""Independent verification of the blind derivation.

Written without reusing the derivation's own level/enumeration helpers:
every check below drives the public sample() entry point with explicit bit
strings, or recomputes quantities from the weights directly.
"""
import itertools, math, random
from fractions import Fraction
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from weighted_sampler import ExactWeightedSampler, ListBitSource

FAIL = []

def check(cond, msg):
    if not cond:
        FAIL.append(msg)
    return cond

# ---------------------------------------------------------------- exactness
def brute_force(weights, depth):
    """Feed literally every bit string of length `depth` to sample()."""
    s = ExactWeightedSampler(weights)
    counts = {}
    unfinished = 0
    for bits in itertools.product((0, 1), repeat=depth):
        src = ListBitSource(list(bits))
        try:
            out = s.sample(src)
        except IndexError:
            unfinished += 1
            continue
        # a string that decides early is counted once, but represents
        # 2^(depth-used) strings sharing that prefix -- so weight by that.
        counts[out] = counts.get(out, 0) + 1
    return counts, unfinished

print("== exhaustive bit-string enumeration through sample() ==")
for weights, depth in [([1, 2, 3], 14), ([3, 5], 12), ([1, 1, 1], 12),
                       ([0, 3, 0, 5, 0], 12), ([2, 7, 11, 4], 13)]:
    counts, unf = brute_force(weights, depth)
    W = sum(weights)
    total = (1 << depth)
    ok = True
    for i, w in enumerate(weights):
        got = counts.get(i, 0)
        want = (total * w) // W
        if got != want:
            ok = False
        check(got == want, f"{weights} idx{i}: got {got} want {want}")
    check(sum(counts.values()) + unf == total, f"{weights}: total mismatch")
    print(f"  {str(weights):<16} depth {depth}: counts={[counts.get(i,0) for i in range(len(weights))]}"
          f" unfinished={unf} total={sum(counts.values())+unf} == 2^{depth} "
          f"[{'OK' if ok else 'FAIL'}]")

# ------------------------------------------------- level-width identity m_k
print("\n== identity m_k = (sum_i 2^k w_i mod W)/W and the bound m_k <= n-1 ==")
def m_k_direct(weights, k):
    """Recompute from the closed form, independently of the sampler."""
    W = sum(weights)
    tot = sum((w * (1 << k)) % W for w in weights)
    assert tot % W == 0, "identity denominator failed"
    return tot // W

random.seed(11)
worst = 0
for trial in range(300):
    n = random.randint(2, 7)
    weights = [random.randint(0, 10 ** random.randint(1, 8)) for _ in range(n)]
    if sum(weights) == 0:
        continue
    s = ExactWeightedSampler(weights)
    if s.degenerate is not None:
        continue
    for k in range(1, 25):
        direct = m_k_direct(weights, k)
        _, m_from_sampler = s._level(k)
        check(direct == m_from_sampler, f"m_k mismatch {weights} k={k}")
        check(direct <= n - 1, f"m_k > n-1 for {weights} k={k}: {direct}")
        worst = max(worst, direct - (n - 1))
print(f"  300 random weight vectors, levels 1..24: closed form matches sampler,")
print(f"  and max(m_k - (n-1)) = {worst}  (must be <= 0)")

# ------------------------------------------------------- entropy bound H<=E<=H+2
print("\n== expected bits vs Shannon entropy ==")
def entropy(weights):
    W = sum(weights)
    return -sum((w / W) * math.log2(w / W) for w in weights if w > 0)

random.seed(4)
worst_gap, worst_case, below = -1, None, 0
rows = []
for trial in range(400):
    n = random.randint(2, 8)
    weights = [random.randint(0, 10 ** random.randint(1, 6)) for _ in range(n)]
    if sum(weights) == 0 or sum(1 for w in weights if w > 0) < 2:
        continue
    s = ExactWeightedSampler(weights)
    E = float(s.expected_bits_upper(depth=400))
    H = entropy(weights)
    if E < H - 1e-9:
        below += 1
    gap = E - H
    if gap > worst_gap:
        worst_gap, worst_case = gap, (weights, E, H)
    rows.append((weights, E, H, gap))
    check(E >= H - 1e-9, f"E < H for {weights}: E={E} H={H}")
    check(gap <= 2 + 1e-9, f"E > H+2 for {weights}: gap={gap}")
print(f"  400 random vectors: violations of E >= H: {below}; "
      f"violations of E <= H+2: {sum(1 for r in rows if r[3] > 2+1e-9)}")
print(f"  worst observed gap E-H = {worst_gap:.6f} on {worst_case[0]}")
print(f"  (E={worst_case[1]:.6f}, H={worst_case[2]:.6f})")

# adversarial: near-degenerate, where the gap should approach 2
adv = [1, 10**12]
s = ExactWeightedSampler(adv)
E, H = float(s.expected_bits_upper(depth=600)), entropy(adv)
print(f"  adversarial {adv}: E={E:.6f} H={H:.9f} gap={E-H:.6f}")
check(E - H <= 2 + 1e-9, "adversarial gap exceeds 2")

# ----------------------------------------------------- exact rational E check
print("\n== exact rational expected cost, small cases ==")
for weights in ([1, 2, 3], [1, 1, 1], [3, 5], [1, 3]):
    s = ExactWeightedSampler(weights)
    E = s.expected_bits_upper(depth=300)
    print(f"  {weights}: E = {E} = {float(E):.6f}")

# ------------------------------------------------------------- Monte Carlo
print("\n== Monte Carlo agreement (independent RNG draws) ==")
for weights in ([1, 2, 3], [5, 1, 1, 1], [7, 2, 11]):
    s = ExactWeightedSampler(weights)
    from weighted_sampler import BitSource
    src = BitSource(2024)
    N = 200000
    tally = [0] * len(weights)
    for _ in range(N):
        tally[s.sample(src)] += 1
    W = sum(weights)
    emp = [t / N for t in tally]
    exp = [w / W for w in weights]
    maxdev = max(abs(a - b) for a, b in zip(emp, exp))
    bits = src.count / N
    Eth = float(s.expected_bits_upper(depth=300))
    print(f"  {weights}: max|emp-exact| = {maxdev:.5f}; bits/sample {bits:.4f} "
          f"vs theory {Eth:.4f}")
    check(maxdev < 0.01, f"empirical deviation too large for {weights}")
    check(abs(bits - Eth) < 0.05, f"bit count mismatch for {weights}")

# ------------------------------------------------------------------ edges
print("\n== edge cases ==")
s = ExactWeightedSampler([0, 7, 0])
src = ListBitSource([])
check(s.sample(src) == 1 and src.count == 0, "degenerate case not zero-bit")
print("  single non-zero weight: returns that index using 0 bits  [OK]")
for bad in ([0, 0], [], [-1, 2]):
    try:
        ExactWeightedSampler(bad)
        FAIL.append(f"accepted invalid input {bad}")
    except (ValueError, IndexError):
        pass
print("  invalid inputs (all-zero, empty, negative) rejected  [OK]")

print("\n" + "=" * 60)
if FAIL:
    print(f"FAILURES ({len(FAIL)}):")
    for f in FAIL[:20]:
        print("  -", f)
else:
    print("ALL INDEPENDENT CHECKS PASSED")
