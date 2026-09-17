"""Post-review corrections to the Case B expected-cost computation.

WHY THIS FILE EXISTS
--------------------
`weighted_sampler.py` is the archived artifact produced in the isolated generating
context. It is deliberately left byte-identical to what was generated, so that
the evidence trail is not rewritten after the fact. Review identified a defect
in it that must not be silently patched there:

    ExactWeightedSampler.expected_bits_upper(depth=K)

is misnamed. It returns

    S_K = sum_{k=0}^{K} m_k 2^{-k},

a truncation of a series with non-negative terms. That is a LOWER partial sum,
never an upper bound, and for a non-terminating residue sequence it is not the
exact infinite expectation either -- exact rational arithmetic makes the partial
sum exact, it does not remove the truncation.

THE CORRECTION
--------------
The proved level-width bound m_k <= n - 1 (Eq. (9) in the manuscript) bounds the
discarded tail:

    sum_{k>K} m_k 2^{-k}  <=  (n-1) sum_{k>K} 2^{-k}  =  (n-1) 2^{-K},

which gives a certified interval containing the true expectation:

    S_K  <=  E[D]  <=  S_K + (n-1) 2^{-K}.

`expected_bits_interval` below returns that interval as exact Fractions.
Every expected-cost figure in the manuscript should be read as the lower
endpoint, with the stated remainder.

Note also that the residue sequence TERMINATES (all residues reach zero)
exactly when every w_i / W has a terminating binary expansion, i.e. when
W / gcd(w_1..w_n, W) is a power of two. In that case the series is finite and
S_K is the exact expectation once K is large enough; `expectation_is_exact`
reports this.

Run:  python3 corrections.py
"""

from fractions import Fraction
from math import gcd
from functools import reduce

from weighted_sampler import ExactWeightedSampler


def expected_bits_interval(weights, K=256):
    """Return (lower, upper) exact Fractions bracketing E[D].

    lower = S_K,  upper = S_K + (n-1) 2^{-K}, by the proved bound m_k <= n-1.
    """
    s = ExactWeightedSampler(weights)
    n = len(weights)
    lower = s.expected_bits_upper(depth=K)          # misnomer in the archive
    if s.degenerate is not None:
        return Fraction(0), Fraction(0)
    upper = lower + Fraction(n - 1, 1 << K)
    return lower, upper


def expectation_is_exact(weights):
    """True iff the binary expansions terminate, so the series is finite."""
    W = sum(weights)
    g = reduce(gcd, [w for w in weights if w > 0] + [W])
    m = W // g
    return m & (m - 1) == 0          # m is a power of two


def _report(weights, K=64):
    lo, hi = expected_bits_interval(weights, K)
    exact = expectation_is_exact(weights)
    width = float(hi - lo)
    tag = "series terminates: value is exact" if exact else \
          f"truncated at K={K}: remainder <= {width:.3e}"
    print(f"  {str(weights):<22} E in [{float(lo):.12f}, {float(hi):.12f}]   {tag}")
    if exact:
        print(f"  {'':22} exact value = {lo}")


if __name__ == "__main__":
    print("Certified intervals for E[D]  (lower partial sum + proved remainder)\n")
    for w in ([3, 5], [1, 3], [1, 2, 3], [1, 1, 1], [2, 7, 11, 4], [1, 10 ** 12]):
        _report(w)

    print("\nMonotonicity check: S_K is non-decreasing in K, confirming it is a")
    print("lower partial sum and not an upper bound.")
    s = ExactWeightedSampler([1, 2, 3])
    prev = None
    for K in (5, 10, 20, 50, 100, 300):
        S = float(s.expected_bits_upper(depth=K))
        flag = "" if prev is None else ("  increasing" if S >= prev else "  DECREASING!")
        print(f"  K={K:<4} S_K = {S:.15f}{flag}")
        prev = S

    print("\nInterval width shrinks geometrically, as (n-1)2^-K predicts:")
    for K in (10, 20, 40, 80):
        lo, hi = expected_bits_interval([2, 7, 11, 4], K)
        print(f"  K={K:<4} width = {float(hi - lo):.3e}")
